"""EXP-005 probability graph on support features, with differentiable memory writes."""
import hashlib
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


def stable_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)


def graph_nodes(feature, mask):
    mask = F.interpolate(mask.float().reshape(1,1,*mask.shape[-2:]), feature.shape[-2:], mode='nearest')
    denominator = F.adaptive_avg_pool2d(mask, (16,16))
    pooled = F.adaptive_avg_pool2d(feature*mask, (16,16))/denominator.clamp_min(1e-8)
    valid = denominator.flatten()>0
    if not valid.any(): raise ValueError('Empty support foreground at feature resolution')
    tokens = pooled.flatten(2)[0,:,valid].T
    with torch.no_grad():
        n=len(tokens)
        similarity=F.normalize(tokens,dim=-1)@F.normalize(tokens,dim=-1).T
        similarity.fill_diagonal_(-float('inf'))
        adjacency=torch.eye(n,device=tokens.device)
        if n>1:
            neighbors=torch.argsort(similarity,dim=1,descending=True,stable=True)[:,:min(8,n-1)]
            adjacency.scatter_(1,neighbors,1)
        adjacency=torch.maximum(adjacency,adjacency.T)
        degree=adjacency.sum(1).rsqrt()
        adjacency=degree[:,None]*adjacency*degree[None,:]
    return tokens,adjacency,valid,mask


class SupportGraph(nn.Module):
    def __init__(self,channels=256):
        super().__init__()
        self.input=nn.Linear(channels,64)
        self.layers=nn.ModuleList([nn.Linear(64,64) for _ in range(2)])
        self.norms=nn.ModuleList([nn.LayerNorm(64) for _ in range(2)])
        self.mu=nn.Linear(64,64)
        self.var=nn.Linear(64,64)
        self.output=nn.Linear(64,channels)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def distribution(self,x,adjacency):
        h=F.gelu(self.input(x))
        for layer,norm in zip(self.layers,self.norms): h=F.gelu(norm(layer(adjacency@h)))
        return self.mu(h),F.softplus(self.var(h))+1e-4

    def forward(self,feature,mask,mode='uncertainty',seed=0,auxiliary=True):
        if mode=='baseline': return feature,feature.new_zeros(()),{}
        if mode not in ('uniform','uncertainty','shuffled'):raise ValueError(mode)
        x,adj,valid,foreground=graph_nodes(feature,mask)
        mu,var=self.distribution(x,adj)
        kl=0.5*(var+mu.square()-1-var.log()).mean()
        gen=torch.Generator(device=feature.device).manual_seed(seed)
        z=mu+torch.randn(mu.shape,generator=gen,device=mu.device)*var.sqrt() if self.training else mu
        consistency=mu.new_zeros(())
        if self.training and auxiliary:
            keep=torch.rand((len(x),1),generator=gen,device=x.device)>=0.1
            aug_mu,aug_var=self.distribution(x*keep,adj)
            consistency=F.mse_loss(F.layer_norm(mu,(64,)),F.layer_norm(aug_mu,(64,)))+0.1*F.mse_loss(var.log(),aug_var.log())
        reliability=torch.exp(-var.mean(-1))
        weight=reliability
        if mode=='uniform':weight=reliability.mean().expand_as(reliability)
        if mode=='shuffled':
            shuffle=torch.Generator(device=feature.device).manual_seed(seed)
            weight=reliability[torch.randperm(len(x),generator=shuffle,device=x.device)]
        values=self.output(z*weight[:,None])
        grid=feature.new_zeros((256,feature.shape[1]))
        grid[valid]=values
        delta=F.interpolate(grid.T.reshape(1,-1,16,16),feature.shape[-2:],mode='bilinear',align_corners=False)
        delta=0.1*torch.tanh(delta)*foreground
        stats={'reliability_mean':reliability.detach().mean(),'reliability_std':reliability.detach().std(unbiased=False),
               'delta_abs_mean':delta.detach().abs().mean(),'delta_abs_max':delta.detach().abs().max(),
               'kl':kl.detach(),'consistency':consistency.detach()}
        return feature+delta,1e-4*kl+0.1*consistency,stats


@dataclass
class SupportCache:
    backbone: object
    original_entries: list
    support_masks: torch.Tensor


class GraphSANSA(nn.Module):
    def __init__(self,sansa):
        super().__init__()
        self.sansa=sansa.requires_grad_(False).eval()
        self.graph=SupportGraph(sansa.sam.hidden_dim).to(sansa.device)

    def train(self,mode=True):
        super().train(mode);self.sansa.eval();return self

    @torch.no_grad()
    def prepare(self,images,masks):
        from util.promptable_utils import rescale_prompt
        if images.shape[:2]!=(1,6) or masks.shape[0]!=5:raise ValueError('Exactly five supports and one query required')
        flat,_,_,sizes=self.sansa._preprocess_visual_features(images,self.sansa.sam.image_size)
        bb=self.sansa._forward_backbone(flat,sizes)
        entries=[]
        for i in range(5):
            prompt=rescale_prompt(masks[i:i+1,None],'mask',sizes[i],self.sansa.sam.image_size)
            entries.append(self.sansa.sam._use_mask_as_output(bb,prompt,i))
        return SupportCache(bb,entries,masks)

    def decode(self,cache,mode='baseline',episode_id='',noise_seed=0,auxiliary=True,retain_memory=False):
        bb=cache.backbone;bank={};loss=bb.vision_feats[-1].new_zeros(());stats=[]
        for i,decoded in enumerate(cache.original_entries):
            features=bb.get_current_feats(i)
            if mode!='baseline':
                raw=bb.get_current_feats_x16(i)
                refined,aux,stat=self.graph(raw,cache.support_masks[i],mode,stable_seed(f'{noise_seed}:{episode_id}:{i}'),auxiliary)
                features=features[:-1]+[refined.flatten(2).permute(2,0,1)]
                loss=loss+aux/5;stats.append(stat)
            # Deliberately outside no_grad: the support graph learns through all five memories.
            mem,pos=self.sansa.sam._encode_new_memory(current_vision_feats=features,feat_sizes=bb.feat_sizes,
                pred_masks_high_res=decoded.high_res_masks,is_mask_from_pts=False)
            if retain_memory and mem.requires_grad:mem.retain_grad()
            bank[i]={'maskmem_features':mem,'maskmem_pos_enc':pos,'pred_masks':decoded.low_res_masks,'obj_ptr':decoded.obj_ptr}
        out=self.sansa._compute_decoder_out_w_mem(bb,5,5,bank)
        logits=F.interpolate(out.masks,bb.orig_size[5],mode='bilinear',align_corners=False)[0,0]
        summary={k:torch.stack([s[k] for s in stats]).mean().item() for k in stats[0]} if stats else {}
        return logits,loss,summary,bank if retain_memory else None
