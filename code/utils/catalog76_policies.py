"""Opt-in actor-local experimental features; never imported by the live campaign."""
import torch
from torch import nn

from utils.networks import MLPNetwork, ACTION_TO_CONTROL
from utils.exploration_topology_policies import local_positions, topology_constants, geometric_statistics, focal_hks
from utils.gsp_features import AGENT_AGENT_SCALE, SIGMA_AGENT_AGENT, SIGMA_AGENT_LANDMARK


FEATURE_RECIPES = {
    'binary_hks': dict(kernel='binary'),
    'rwse246': dict(descriptor='rwse246'),
    'split_hks': dict(split=True),
    'pair_hks': dict(topology='pairs'),
    'hypergraph_hks': dict(topology='hypergraph'),
    'radius_hks': dict(support='radius'),
    'mutual_hks': dict(support='mutual'),
    'landmark_top2_hks': dict(support='landmark_top2'),
    'weak_long_hks': dict(support='weak_long'),
    'compact_hks': dict(kernel='compact'),
    'relation_scales_hks': dict(topology='6all', relation_scales=True),
    'adaptive_hks': dict(kernel='adaptive'),
    'contact_weight_hks': dict(topology='6aal', kernel='contact'),
    'adjacency_readout': dict(descriptor='adjacency'),
    'coordinate_filter': dict(descriptor='signal', signal='coordinates'),
    'coverage_filter': dict(descriptor='signal', signal='coverage'),
    'crowding_filter': dict(descriptor='signal', signal='crowding'),
    'affinity_filter': dict(descriptor='signal', signal='affinity'),
    'advantage_filter': dict(descriptor='signal', signal='advantage'),
    'velocity_filter': dict(descriptor='signal', signal='velocity'),
    'degree_stats': dict(descriptor='degree'),
    'mask_filter': dict(descriptor='signal', signal='mask'),
    'effective_resistance': dict(descriptor='resistance'),
    'spectrum_moments': dict(descriptor='spectrum'),
    'static_energy': dict(descriptor='energy'),
    'rayleigh_energy': dict(descriptor='rayleigh'),
    'cross_energy': dict(descriptor='cross'),
    'wavelet_signal': dict(descriptor='wavelet'),
    'action_absolute': dict(descriptor='action', mode='absolute', motion='all'),
    'action_current_delta': dict(descriptor='action', mode='current_delta', motion='all'),
    'action_self_contact': dict(descriptor='action', mode='delta', motion='self'),
    'action_hypotheses': dict(descriptor='action', mode='delta', motion='hypotheses'),
    'action_horizon3': dict(descriptor='action', mode='delta', motion='horizon3'),
    'layernorm_hks': dict(layernorm=True),
}


def graph(raw, config):
    topology = config.get('topology', '6al')
    points = local_positions(raw, '6al' if topology in ('pairs', 'hypergraph') else topology)
    if topology == 'pairs':
        a, l = points[:, :3], points[:, 3:]
        # Nine pairs, with a four-dimensional (agent, landmark) geometry.
        points = torch.cat((a[:, :, None].expand(-1, -1, 3, -1),
                            l[:, None].expand(-1, 3, -1, -1)), -1).reshape(-1, 9, 4)
        index = torch.arange(9, device=raw.device)
        mask = ((index[:, None] // 3 == index[None] // 3) |
                (index[:, None] % 3 == index[None] % 3)).to(raw.dtype)
        mask.fill_diagonal_(0)
        sigma = torch.full_like(mask, .6)
    else:
        mask, sigma = topology_constants('6al' if topology == 'hypergraph' else topology)
        mask, sigma = mask.to(raw), sigma.to(raw)
    squared = (points[:, :, None] - points[:, None, :]).square().sum(-1)
    distance = squared.clamp_min(1e-24).sqrt()
    if config.get('relation_scales'):
        sigma = sigma.clone()
        sigma[:3, :3] = .6
        sigma[3:, 3:] = 1.2
    kernel = config.get('kernel', 'gaussian')
    if kernel == 'binary':
        weights = torch.ones_like(squared)
    elif kernel == 'adaptive':
        scale = distance[:, mask.bool()].median(-1).values.clamp_min(.05)
        weights = torch.exp(-squared / (2 * scale[:, None, None].square()))
    elif kernel == 'compact':
        r = (distance / 1.2).clamp(max=1)
        weights = (1-r).pow(4) * (1+4*r)
    else:
        weights = torch.exp(-squared / (2 * sigma.square()))
        if kernel == 'contact':
            aa = torch.zeros_like(mask)
            aa[:3, :3] = 1
            weights = weights * (1-aa) + torch.sigmoid((.30-distance)/.05) * aa
    weights = weights * mask
    support = config.get('support', 'full')
    if support in ('mutual', 'landmark_top2'):
        if support == 'mutual':
            nearest = squared.masked_fill(mask == 0, float('inf')).topk(2, largest=False).indices
            directed = torch.zeros_like(weights).scatter_(-1, nearest, 1) * mask
            weights = weights * directed * directed.transpose(-1, -2)
        else:
            selected = squared[:, :3, 3:].topk(2, dim=1, largest=False).indices
            kept = torch.zeros_like(weights[:, :3, 3:]).scatter_(1, selected, 1)
            support_mask = torch.zeros_like(weights)
            support_mask[:, :3, 3:] = kept
            support_mask[:, 3:, :3] = kept.transpose(-1, -2)
            weights = weights * support_mask
    elif support == 'radius':
        weights = weights * (distance <= 1.)
    elif support == 'weak_long':
        weights = weights * torch.where(distance <= 1., 1., .05)
    if topology == 'hypergraph':
        # Three landmark-centred hyperedges. H=[AL affinities; landmark identity].
        h = torch.cat((weights[:, :3, 3:], torch.eye(3, device=raw.device).expand(raw.shape[0], -1, -1)), 1)
        weights = (h / h.sum(1, keepdim=True).clamp_min(1e-12)) @ h.transpose(-1, -2)
        weights = weights * (1-torch.eye(6, device=raw.device))
    return weights


def operators(w):
    degree = w.sum(-1)
    inverse = degree.clamp_min(1e-12).rsqrt()
    s = inverse[:, :, None] * w * inverse[:, None, :]
    identity = torch.eye(w.shape[-1], device=w.device, dtype=w.dtype)
    return s, identity-s, torch.diag_embed(degree)-w


def node_signal(raw, w, kind):
    p = local_positions(raw, '6al')
    al = (p[:, :3, None]-p[:, None, 3:]).square().sum(-1).sqrt()
    signal = raw.new_zeros((raw.shape[0], 6, 3))
    if kind == 'coordinates':
        signal[:, :, :2] = p
        signal[:, 3:, 2] = 1
    elif kind == 'coverage':
        signal[:, 3:, 0] = al.amin(1)
    elif kind == 'crowding':
        aa = (p[:, :3, None]-p[:, None, :3]).square().sum(-1)
        proximity = torch.exp(-aa/(2*.8**2))*(1-torch.eye(3, device=raw.device))
        signal[:, :3, 0] = proximity.sum(-1)
    elif kind == 'affinity':
        signal[:, :3] = torch.exp(-al.square()/(2*.6**2))
    elif kind == 'advantage':
        near = al.sort(1).values
        signal[:, 3:, 0] = near[:, 1]-near[:, 0]
    elif kind == 'velocity':
        direction = p[:, 3:] / p[:, 3:].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        signal[:, 3:, 0] = (direction * raw[:, None, :2]).sum(-1)
    elif kind == 'mask':
        signal[:, :, 0] = 1  # All relative positions observed in this scenario.
        signal[:, 0, 1] = 1  # Only focal velocity is observed.
        signal[:, 3:, 2] = 1
    else:
        raise ValueError(kind)
    return signal


def pair_energy(w, x):
    return .5 * (w[..., None] * (x[:, :, None]-x[:, None, :]).square()).sum((1, 2))


def action_quantities(raw, motion):
    p = local_positions(raw, '6al')
    agents, landmarks = p[:, :3], p[:, 3:]
    controls = raw.new_tensor(ACTION_TO_CONTROL)
    candidates = agents[:, None].expand(-1, 5, -1, -1).clone()
    velocity = torch.zeros_like(candidates)
    velocity[:, :, 0] = raw[:, None, :2]
    horizon = 0 if motion == 'current' else 3 if motion == 'horizon3' else 1
    for _ in range(horizon):
        diff = candidates[:, :, :, None]-candidates[:, :, None, :]
        distance = diff.square().sum(-1).clamp_min(1e-24).sqrt()
        penetration = torch.logaddexp(torch.zeros_like(distance), (.3-distance)/.001)*.001
        drift = (penetration[..., None]*diff/distance[..., None]).sum(-2)
        if motion == 'self':
            drift = drift * raw.new_tensor([1, 0, 0])[None, None, :, None]
        velocity = velocity * .75 + drift / .1
        velocity[:, :, 0] += controls[None] * .5
        candidates = candidates + velocity * .1
    if motion == 'hypotheses':
        # Symmetric fixed common-neighbour drift hypotheses, NOT observed velocities.
        hypotheses = raw.new_tensor([[0, 0], [.025, 0], [-.025, 0], [0, .025], [0, -.025]])
        candidates = candidates[:, None].expand(-1, 5, -1, -1, -1).clone()
        candidates[:, :, :, 1:] += hypotheses[None, :, None, None, :]
        candidates = candidates.flatten(0, 1)
        landmarks = landmarks[:, None].expand(-1, 5, -1, -1).flatten(0, 1)
    aa = (candidates[:, :, :, None]-candidates[:, :, None, :]).square().sum(-1)
    al = (candidates[:, :, :, None]-landmarks[:, None, None]).square().sum(-1)
    eye = torch.eye(3, device=raw.device)
    waa = AGENT_AGENT_SCALE*torch.exp(-aa/(2*SIGMA_AGENT_AGENT**2))*(1-eye)
    wal = torch.exp(-al/(2*SIGMA_AGENT_LANDMARK**2))
    cov = al.amin(2)
    crowd = AGENT_AGENT_SCALE*torch.exp(-(aa+eye*1e12).amin(-1)/(2*SIGMA_AGENT_AGENT**2))
    quantities = torch.stack((cov.clamp_min(1e-24).sqrt().sum(-1), waa.sum((-1, -2))*.5,
        (wal*cov[:, :, None]).sum((-1, -2)),
        (wal*crowd.square()[:, :, :, None]).sum((-1, -2))+
        .5*(waa*(crowd[:, :, :, None]-crowd[:, :, None, :]).square()).sum((-1, -2))), -1)
    return quantities.reshape(raw.shape[0], 5, 5, 4).mean(1) if motion == 'hypotheses' else quantities


def features(raw, config):
    if raw.ndim != 2 or raw.shape[-1] != 18 or raw.dtype != torch.float32:
        raise ValueError('Expected one actor local float32 observation per row [M,18]')
    descriptor = config.get('descriptor', 'hks')
    if descriptor == 'action':
        q = action_quantities(raw, config['motion'])
        delta = q-q[:, :1]
        if config['mode'] == 'absolute':
            return q.flatten(1)
        if config['mode'] == 'current_delta':
            current = action_quantities(raw, 'current')[:, 0]
            return torch.cat((current, delta.flatten(1)), -1)
        return delta.flatten(1)
    if config.get('split'):
        return torch.cat((focal_hks(graph(raw, {'topology':'3aa'}), raw.new_tensor([.5,1,2])),
                          focal_hks(graph(raw, {}), raw.new_tensor([.5,1,2]))), -1)
    w = graph(raw, config)
    s, lap, combinatorial = operators(w)
    if descriptor == 'rwse246':
        s2 = s@s
        s4 = s2@s2
        s6 = s4@s2
        return torch.stack((s2[:,0,0],s4[:,0,0],s6[:,0,0]), -1)
    if descriptor == 'hks':
        return focal_hks(w, raw.new_tensor([.5,1,2]))
    if descriptor == 'adjacency':
        return w[:, 0, 3:6]
    if descriptor == 'degree':
        d = w.sum(-1)
        return torch.stack((d[:, 0], d.mean(-1), d.std(-1, unbiased=False)), -1)
    if descriptor == 'signal':
        x = node_signal(raw, w, config['signal'])
        y = s @ x
        return torch.stack((y[:, 0, 0], (s@y)[:, 0, 0], y[:, 3:, 0].mean(-1)), -1) if config['signal'] not in ('coordinates','affinity','mask') else y[:, 0]
    if descriptor == 'spectrum':
        eig = torch.linalg.eigvalsh(lap).clamp(0, 2)
        return torch.stack((eig[:, 1], eig.square().mean(-1), eig.pow(3).mean(-1)), -1)
    if descriptor == 'resistance':
        values, vectors = torch.linalg.eigh(combinatorial)
        positive = values > 1e-6
        inverse = torch.where(positive, values.clamp_min(1e-6).reciprocal(), 0.)
        difference = vectors[:, :1]-vectors[:, 3:6]
        resistance = (difference.square()*inverse[:, None]).sum(-1)
        disconnected = (difference.square()*(~positive)[:, None]).sum(-1) > 1e-5
        # Missing resistance is encoded as zero PLUS a validity mask, never a finite physical claim.
        return torch.cat((resistance.masked_fill(disconnected, 0), (~disconnected).float()), -1)
    x = node_signal(raw, w, 'affinity')
    if descriptor == 'cross':
        cross = x.transpose(-1, -2) @ combinatorial @ x
        return torch.stack((cross[:, 0, 1], cross[:, 0, 2], cross[:, 1, 2]), -1)
    if descriptor == 'wavelet':
        heat = torch.matrix_exp(-lap)
        response = (heat-heat@heat) @ x
        return response[:, 0]
    if descriptor == 'energy':
        coverage = node_signal(raw, w, 'coverage')[:, :, :1]
        crowding = node_signal(raw, w, 'crowding')[:, :, :1]
        return torch.cat((pair_energy(w, coverage), pair_energy(w, crowding), pair_energy(w, x).sum(-1, keepdim=True)), -1)
    if descriptor == 'rayleigh':
        filtered = torch.matrix_exp(-lap) @ x
        return pair_energy(w, filtered)/filtered.square().sum(1).clamp_min(1e-12)
    raise ValueError(descriptor)


class FeaturePolicy(nn.Module):
    recipe = 'split_hks'
    control = False

    def __init__(self, input_dim, out_dim, **kwargs):
        super().__init__()
        if input_dim != 18:
            raise ValueError('Raw actor input must remain 18D')
        self.config = FEATURE_RECIPES[self.recipe]
        with torch.no_grad():
            self.feature_dim = features(torch.zeros(1, 18), self.config).shape[-1]
        self.actor_input_dim = 18+self.feature_dim
        self.mlp = MLPNetwork(self.actor_input_dim, out_dim, **kwargs)
        self.normalizer = nn.LayerNorm(self.feature_dim) if self.config.get('layernorm') else nn.Identity()
        self.last_raw_input_shape = self.last_augmented_input_shape = None

    def augment_observation(self, raw):
        with torch.no_grad():
            if self.control:
                if self.config.get('descriptor') == 'action':
                    feature = features(raw, self.config).reshape(raw.shape[0], -1, 4)
                    feature = (feature * raw.new_tensor([1, 1, 0, 0])).flatten(1)
                else:
                    base = geometric_statistics(raw)
                    feature = base.repeat(1, (self.feature_dim+2)//3)[:, :self.feature_dim]
            else:
                feature = features(raw, self.config)
        augmented = torch.cat((raw, self.normalizer(feature)), -1)
        self.last_raw_input_shape = tuple(raw.shape)
        self.last_augmented_input_shape = tuple(augmented.shape)
        return augmented

    def forward(self, raw):
        return self.mlp(self.augment_observation(raw))


class EncoderPolicy(nn.Module):
    mode = 'sets'
    control = False

    def __init__(self, input_dim, out_dim, hidden_dim=64, **kwargs):
        super().__init__()
        if input_dim != 18:
            raise ValueError('Raw actor input must remain 18D')
        self.actor_input_dim = 18
        self.mlp = MLPNetwork(18, out_dim, hidden_dim=hidden_dim, **kwargs)
        self.encoder = nn.Linear(5, 32)
        self.edge = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))
        self.query = nn.Linear(5, 32, bias=False)
        self.out = nn.Linear(96, out_dim)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, raw):
        p = local_positions(raw, '6al')
        flags = raw.new_zeros((raw.shape[0],6,3))
        flags[:, :3, 0] = 1
        flags[:, 3:, 1] = 1
        flags[:, 0, 2] = 1
        x = torch.cat((p, flags), -1)
        h = torch.relu(self.encoder(x))
        if self.mode in ('gcn', 'learned', 'directed'):
            w = graph(raw, {'topology':'6aal'})
            if self.mode in ('learned', 'directed'):
                pair = torch.cat((x[:, :, None].expand(-1,-1,6,-1),x[:,None].expand(-1,6,-1,-1)), -1)
                gate = torch.sigmoid(self.edge(pair).squeeze(-1))
                if self.mode == 'learned':
                    gate = (gate+gate.transpose(-1,-2))*.5
                w = w*gate
            if not self.control:
                h = h + (w/w.sum(-1,keepdim=True).clamp_min(1e-12))@h
        if self.mode == 'attention':
            q = self.query(x[:, :1])
            weight = torch.softmax((q*h).sum(-1)/(32**.5), -1)
            pooled = (weight[...,None]*h).sum(1) if not self.control else h.mean(1)
        else:
            pooled = h.mean(1)
        return self.mlp(raw)+self.out(torch.cat((h[:,0],pooled,h[:,3:].mean(1)), -1))


class VectorProjectionPolicy(nn.Module):
    control = False

    def __init__(self, input_dim, out_dim, hidden_dim=64, **kwargs):
        super().__init__()
        if input_dim != 18 or out_dim != 5:
            raise ValueError('Vector branch requires raw18 and original five actions')
        self.actor_input_dim = 18
        self.mlp = MLPNetwork(18, 5, hidden_dim=hidden_dim, **kwargs)
        self.scalar = nn.Sequential(nn.Linear(3,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.normal_(self.scalar[-1].weight,std=1e-3)
        nn.init.zeros_(self.scalar[-1].bias)

    def graph_residual(self, raw):
        p = local_positions(raw,'6al')
        kinds = raw.new_zeros((raw.shape[0],6,2))
        kinds[:,:3,0] = 1
        kinds[:,3:,1] = 1
        invariant = torch.cat((p.square().sum(-1,keepdim=True).clamp_min(1e-24).sqrt(),kinds),-1)
        weights = torch.tanh(self.scalar(invariant))
        vectors = p if self.control else operators(graph(raw,{'topology':'6aal'}))[0]@p
        message = (weights*vectors).sum(1)
        return message @ raw.new_tensor(ACTION_TO_CONTROL).T

    def forward(self, raw):
        return self.mlp(raw)+self.graph_residual(raw)


def register_policies():
    from utils.agents import POLICY_TYPES
    for name in FEATURE_RECIPES:
        for control in (False, True):
            key = 'catalog_'+name+('_control' if control else '')
            POLICY_TYPES[key] = type(key, (FeaturePolicy,), {'recipe':name, 'control':control})
    for mode in ('sets','gcn','learned','directed','attention'):
        for control in (False, True):
            key = 'catalog_encoder_'+mode+('_control' if control else '')
            POLICY_TYPES[key] = type(key, (EncoderPolicy,), {'mode':mode, 'control':control})
    for control in (False, True):
        key = 'catalog_vector_projection'+('_control' if control else '')
        POLICY_TYPES[key] = type(key,(VectorProjectionPolicy,),{'control':control})
