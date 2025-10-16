import gzip, json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import rdChemReactions

    
with gzip.open('data/reactions_infos.json.gz', 'r') as f:
    reactions_infos = json.loads(f.read().decode('utf-8'))["reactions"]

building_blocks = pd.read_csv(f'data/matched_building_blocks.csv.gz', compression='gzip')['SMILES'].tolist()
bb_2_id = {building_blocks[i]: i for i in range(len(building_blocks))}
bb_fps = np.load('data/building_blocks_fp.npy')
print(len(building_blocks), len(list(bb_2_id.keys())), bb_fps.shape)
            
def get_morgen_fingerprint(smiles, nBits=4096):
    if smiles is None:
        return np.zeros(nBits).reshape((-1, )).tolist()
    else:
        mol = Chem.MolFromSmiles(smiles)
        fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp_vec, features)
        return features.reshape((-1, )).tolist()
    
def get_reaction_mask(smiles, templates):
    if smiles is None:
        return [0] * len(templates)
    else:
        mol = Chem.MolFromSmiles(smiles)
        template_mask = []
        for template in templates:
            rxn = AllChem.ReactionFromSmarts(template)
            rdChemReactions.ChemicalReaction.Initialize(rxn)
            result = rxn.IsMoleculeReactant(mol)
            template_mask.append(int(result))
        return template_mask

def  check_match_smarts(mol, rxn):
    r1_smarts, r2_smarts = rxn.split('>')[0].split('.')
    match_first = mol.HasSubstructMatch(Chem.MolFromSmarts(r1_smarts))
    match_second = mol.HasSubstructMatch(Chem.MolFromSmarts(r2_smarts))
    if match_first and (not match_second):
        return 0
    elif (not match_first) and match_second:
        return 1
    else:
        return 2    
   
def get_second_reactant(tid, rxn, query_emb):
    available_frags = get_available_list(tid, rxn)  
    available_ids = [bb_2_id[available_frags[i]] for i in range(len(available_frags))]
    if len(available_ids) == 0:
        print('length of available is 0.')
        return None
    else:
        temp_emb  = bb_fps[available_ids]  
        ind = search_with_tanimoto(query_emb, temp_emb)        
        return building_blocks[available_ids[ind[0]]]
    
def search_with_tanimoto(emb, avali_embs, ntop=1):
    emb = np.squeeze(emb)
    sims = []
    for aemb in avali_embs:
        tanimoto_smi = np.sum(emb * aemb) / (np.sum(emb) + np.sum(aemb) - np.sum(emb * aemb))
        sims.append(tanimoto_smi)
    
    sims = np.array(sims)
    top_simi_idx = sims.argsort()[::-1][0:ntop]
    
    return top_simi_idx

def get_available_list(tid, rxn):
    for reac_info in reactions_infos:
        if reac_info['smirks'] == rxn:
            if tid == 0:
                available_frag_list = reac_info['available_reactants'][1]
            elif tid == 1:
                available_frag_list = reac_info['available_reactants'][0]
            else:
                available_frag_list = []
        
    return available_frag_list
