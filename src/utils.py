import torch
import numpy as np

from src.constants import NT_DICT


_SHARP_LOOP_DIST_THRESHOLD = 4
def _generate_sharp_loop_mask(seq_len):
    mask = np.eye(seq_len, k=0, dtype=bool)
    for i in range(1, _SHARP_LOOP_DIST_THRESHOLD):
        mask = mask + np.eye(seq_len, k=i, dtype=bool) + np.eye(seq_len, k=-i, dtype=bool)

    return mask

CANONICAL_PAIRS = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']
def _generate_canonical_pairs_mask(seq: str):
    seq = seq.replace('T', 'U')

    mask = np.zeros((len(seq), len(seq)), dtype=bool)
    for i, nt_i in enumerate(seq):
        for j, nt_j in enumerate(seq):
            if f'{nt_i}{nt_j}' in CANONICAL_PAIRS:
                mask[i, j] = True

    return mask

def _clean_sec_struct(sec_struct: np.array, probs: np.array):
    clean_sec_struct = np.copy(sec_struct)
    tmp_probs = np.copy(probs)
    tmp_probs[sec_struct < 1] = 0.0

    while np.sum(tmp_probs > 0.0) > 0:
        i, j = np.unravel_index(np.argmax(tmp_probs, axis=None), tmp_probs.shape)

        tmp_probs[i, :] = tmp_probs[j, :] = 0.0
        clean_sec_struct[i, :] = clean_sec_struct[j, :] = 0

        tmp_probs[:, i] = tmp_probs[:, j] = 0.0
        clean_sec_struct[:, i] = clean_sec_struct[:, j] = 0

        clean_sec_struct[i, j] = clean_sec_struct[j, i] = 1

    return clean_sec_struct

def prob_mat_to_sec_struct(probs: np.array, seq: str, threshold: float = 0.5, allow_nc_pairs: bool = False, allow_sharp_loops: bool = False):
    assert np.all(np.isclose(probs, np.transpose(probs))), "Probability matrix must be symmetric!"
    seq_len = probs.shape[-1]

    allowed_pairs_mask = np.logical_not(np.eye(seq_len, dtype=bool))

    if not allow_sharp_loops:  
        # Prevent pairings that would cause sharp loops
        allowed_pairs_mask = np.logical_and(allowed_pairs_mask, ~_generate_sharp_loop_mask(seq_len))

    if not allow_nc_pairs:
        # Prevent non-canonical pairings
        allowed_pairs_mask = np.logical_and(allowed_pairs_mask, _generate_canonical_pairs_mask(seq))

    probs[~allowed_pairs_mask] = 0.0

    sec_struct = np.greater(probs, threshold).astype(int)
    sec_struct = _clean_sec_struct(sec_struct, probs)

    return sec_struct

def outer_concat(t1: torch.Tensor, t2: torch.Tensor):
    # t1, t2: shape = B x L x E
    assert t1.shape == t2.shape, f"Shapes of input tensors must match! ({t1.shape} != {t2.shape})"

    seq_len = t1.shape[1]
    a = t1.unsqueeze(-2).expand(-1, -1, seq_len, -1)
    b = t2.unsqueeze(-3).expand(-1, seq_len, -1, -1)

    return torch.concat((a, b), dim=-1)


def postprocessing(preds, masks):
    """Postprocessing function using viable pairing mask.
    Inputs are batches of size [B, N, N]"""
    if masks is not None:
        preds = preds.multiply(masks)

    y_pred_mask_triu = torch.triu(preds)
    y_pred_mask_max = torch.zeros_like(preds)
    for k in range(preds.shape[0]):
        y_pred_mask_max_aux = torch.zeros_like(y_pred_mask_triu[k, :, :])

        val, ind = y_pred_mask_triu[k, :, :].max(dim=0)
        y_pred_mask_max[k, ind[val > 0], val > 0] = val[val > 0]

        val, ind = y_pred_mask_max[k, :, :].max(dim=1)
        y_pred_mask_max_aux[val > 0, ind[val > 0]] = val[val > 0]

        ind = torch.where(y_pred_mask_max[k, :, :] != y_pred_mask_max_aux)
        y_pred_mask_max[k, ind[0], ind[1]] = 0

        y_pred_mask_max[k] = torch.triu(y_pred_mask_max[k]) + torch.triu(
            y_pred_mask_max[k]
        ).transpose(0, 1)
    return y_pred_mask_max


def mat2bp(x):
    """Get base-pairs from conection matrix [N, N]. It uses upper
    triangular matrix only, without the diagonal. Positions are 1-based. """
    ind = torch.triu_indices(x.shape[0], x.shape[1], offset=1)
    pairs_ind = torch.where(x[ind[0], ind[1]] > 0)[0]

    pairs_ind = ind[:, pairs_ind].T
    # remove multiplets pairs
    multiplets = []
    for i, j in pairs_ind:
        ind = torch.where(pairs_ind[:, 1]==i)[0]
        if len(ind)>0:
            pairs = [bp.tolist() for bp in pairs_ind[ind]] + [[i.item(), j.item()]]
            best_pair = torch.tensor([x[bp[0], bp[1]] for bp in pairs]).argmax()
                
            multiplets += [pairs[k] for k in range(len(pairs)) if k!=best_pair]   
            
    pairs_ind = [[bp[0]+1, bp[1]+1] for bp in pairs_ind.tolist() if bp not in multiplets]
 
    return pairs_ind


def pair_strength(pair):
    if "G" in pair and "C" in pair:
        return 3
    if "A" in pair and "U" in pair:
        return 2
    if "G" in pair and "U" in pair:
        return 0.8

    if pair[0] in NT_DICT and pair[1] in NT_DICT:
        n0, n1 = NT_DICT[pair[0]], NT_DICT[pair[1]]
        # Possible pairs with other bases
        if ("G" in n0 and "C" in n1) or ("C" in n0 and "G" in n1):
            return 3
        if ("A" in n0 and "U" in n1) or ("U" in n0 and "A" in n1):
            return 2
        if ("G" in n0 and "U" in n1) or ("U" in n0 and "G" in n1):
            return 0.8

    return 0


def valid_mask(seq, L):
    """Create a NxN mask with valid canonic pairings."""

    seq = seq.upper().replace("T", "U")  # rna
    mask = torch.zeros(L, L, dtype=torch.float32)
    for i in range(len(seq)):
        for j in range(len(seq)):
            if np.abs(i - j) > 3:  # nt that are too close are invalid
                if pair_strength([seq[i], seq[j]]) > 0:
                    mask[i, j] = 1
                    mask[j, i] = 1
    return mask

def get_embed_dim(loader):
    # grab an element from the loader, which is represented by a dictionary with keys
    # `seq_ids`, `seq_embs_pad`, `contacts`, `Ls`
    batch_elem = next(iter(loader))
    # query for `seq_embs_pad` key (containing the embedding representations of all the sequences in the batch)
    # whose size will be batch_size x L x d
    return batch_elem["seq_embs_pad"].shape[2]
