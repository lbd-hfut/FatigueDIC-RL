"""Native-torch graph message passing + causal graph-level GRU.

No import in the numerical/core paths; no PyG or SciPy autograd bridge.
"""
from dataclasses import dataclass
import time
import torch
from torch import nn
from .parameter_head import ParameterHead
from .uncertainty_head import UncertaintyHead
from .graph_fusion import torch_shape_aggregate


@dataclass(frozen=True)
class NetworkConfig:
    parameter_names: tuple[str,...]=('d0','A')
    hidden_channels: int=64
    graph_layers: int=3
    recurrent: bool=True
    physics: bool=True
    fusion: str='shape'
    uncertainty: bool=True

    def __post_init__(self):
        if self.fusion not in ('shape','global') or not self.parameter_names or self.hidden_channels<4 or self.graph_layers<1:
            raise ValueError('invalid neural architecture contract')


def pack_graph_frames(frames, *, device='cpu', physics=True):
    dic_offset=fe_offset=0;data={k:[] for k in ('df','de','dx','ff','fe','fx','ids','weights','db','fb','global','previous')}
    for i,g in enumerate(frames):
        df=g.dic.features.copy();ff=g.fe.features.copy();glob=g.global_features.copy()
        if not physics:df[:,4:10]=0.;ff[:,-2:]=0.;glob[6:18]=0.
        for k,v in {'df':df,'de':g.dic.edge_index+dic_offset,'dx':g.dic.coordinates_mm,'ff':ff,'fe':g.fe.edge_index+fe_offset,'fx':g.fe.coordinates_mm,'ids':g.mapping_node_ids+fe_offset,'weights':g.mapping_weights,'db':torch.full((len(df),),i),'fb':torch.full((len(ff),),i),'global':glob[None,:],'previous':g.previous_normalized[None,:]}.items():
            data[k].append(torch.as_tensor(v.copy() if hasattr(v,'copy') else v,device=device,dtype=torch.long if k in ('de','fe','ids','db','fb') else torch.float32))
        dic_offset+=len(df);fe_offset+=len(ff)
    result={k:torch.cat(v,dim=1 if k in ('de','fe') else 0) for k,v in data.items()};result['batch_size']=len(frames)
    if all(g.training_whitened_measured is not None for g in frames):
        for key,attribute in [('white_measured','training_whitened_measured'),('white_basis','training_whitened_affine_basis')]:
            result[key]=torch.as_tensor(__import__('numpy').concatenate([getattr(g,attribute) for g in frames]),device=device,dtype=torch.float32)
    return result


def mean_pool(values,batch,count,mask=None):
    if mask is None:mask=torch.ones(len(values),device=values.device,dtype=values.dtype)
    else:mask=mask.to(values.dtype)
    out=values.new_zeros((count,values.shape[1]));mass=values.new_zeros((count,1))
    out.index_add_(0,batch,values*mask[:,None]);mass.index_add_(0,batch,mask[:,None])
    return out/mass.clamp_min(1.)


class MessageLayer(nn.Module):
    def __init__(self,hidden):
        super().__init__();self.message=nn.Sequential(nn.Linear(2*hidden+2,hidden),nn.SiLU(),nn.Linear(hidden,hidden));self.update=nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden));self.norm=nn.LayerNorm(hidden)

    def forward(self,h,x,edges):
        src,dst=edges;messages=self.message(torch.cat([h[src],h[dst],x[src]-x[dst]],dim=-1));a=torch.zeros_like(h);mass=h.new_zeros((len(h),1));a.index_add_(0,dst,messages);mass.index_add_(0,dst,h.new_ones((len(src),1)))
        return self.norm(h+self.update(torch.cat([h,a/mass.clamp_min(1.)],dim=-1)))


class GraphEncoder(nn.Module):
    def __init__(self,inputs,hidden,layers):
        super().__init__();self.input=nn.Sequential(nn.Linear(inputs,hidden),nn.SiLU());self.layers=nn.ModuleList([MessageLayer(hidden) for _ in range(layers)])

    def forward(self,features,coordinates,edges):
        h=self.input(features)
        for layer in self.layers:h=layer(h,coordinates,edges)
        return h


class GraphRecurrentEstimator(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or NetworkConfig();c=self.config;h=c.hidden_channels;k=len(c.parameter_names)
        self.dic=GraphEncoder(19,h,c.graph_layers);self.fe=GraphEncoder(9,h,c.graph_layers)
        self.fusion=nn.Sequential(nn.Linear(2*h,h),nn.SiLU(),nn.Linear(h,h));self.encode=nn.Sequential(nn.Linear(2*h+28+k,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU())
        self.gru=nn.GRUCell(h,h) if c.recurrent else None;self.parameter_head=ParameterHead(h,k);self.uncertainty_head=UncertaintyHead(h,k)

    def step(self,data,hidden=None):
        c=self.config;dh=self.dic(data['df'],data['dx'],data['de']);fh=self.fe(data['ff'],data['fx'],data['fe']);b=data['batch_size']
        if c.fusion=='shape':
            scatter,active=torch_shape_aggregate(dh,data['ids'],data['weights'],len(fh));fh=fh+self.fusion(torch.cat([fh,scatter],dim=-1))*active[:,None]
        prior_feature=data['previous'] if c.physics or not c.recurrent else torch.zeros_like(data['previous'])
        gp=torch.cat([mean_pool(dh,data['db'],b),mean_pool(fh,data['fb'],b),data['global'],prior_feature],dim=-1);encoded=self.encode(gp)
        if self.gru is not None:
            if hidden is None:hidden=encoded.new_zeros((b,c.hidden_channels))
            hidden=self.gru(encoded,hidden)
        else:hidden=encoded
        mean,delta=self.parameter_head(hidden,data['previous']);covariance,root=self.uncertainty_head(hidden)
        return {'normalized_mean':mean,'delta_normalized':delta,'covariance_normalized':covariance,'cholesky':root,'hidden':hidden}

    def forward(self,sequence):
        h=None;outputs=[]
        for data in sequence:
            o=self.step(data,h);h=o['hidden'];outputs.append(o)
        return outputs

    @staticmethod
    def clone_hidden(hidden):return None if hidden is None else hidden.clone()

    def observation_update(self,frame,predictor,previous_normalized,previous_cycle,hidden):
        """Injected observation-only backend; torch stays outside inference contracts."""
        from .graph_features import build_graph_frame
        graph=build_graph_frame(frame,previous_normalized,predictor=predictor,physics=self.config.physics,previous_cycle=previous_cycle)
        t=time.perf_counter();device=next(self.parameters()).device
        with torch.no_grad():result=self.step(pack_graph_frames([graph],device=device,physics=self.config.physics),hidden)
        return {'normalized_mean':result['normalized_mean'][0].cpu().numpy().astype(float),
                'covariance_normalized':result['covariance_normalized'][0].cpu().numpy().astype(float) if self.config.uncertainty else None,
                'hidden':result['hidden'].detach(),'network_seconds':time.perf_counter()-t}
