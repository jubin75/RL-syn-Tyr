import os
import torch 
import json
import numpy as np
from utils import *
from model import GetTemplate, GetAction
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

with open('data/starting_building_blocks.txt', 'r') as f:
    start_building_blocks = [l.strip() for l in f.readlines()]
    print('start_building_blocks =', len(start_building_blocks))

with open('data/rxn_set.txt', 'r') as f:
    templates = [l.strip() for l in f.readlines()]

rxn_model = GetTemplate(state_dim=256, t_dim=len(templates))
rxn_model.load('ckpt/rxn_model.pth')
rxn_model.eval()
reagent_model = GetAction(state_dim=256, t_dim=len(templates), act_dim=256)
reagent_model.load('ckpt/react_model.pth')
reagent_model.eval()

pdbqt_file = 'data/cxcr4/3odu.pdbqt' 
box_center = (12.482525, -5.691434, 66.46724)
box_size = (18.484001, 22.031, 19.056)

max_steps = 3
syn_plan = dict()
reac_num = 0
for i, sbb in enumerate(start_building_blocks[:30]):
    print(f'\n++++++++++++++++++++ ref smi {i}:', sbb)
    step = 0
    one_step_reaction = []
    rt1 = sbb
    while step < max_steps:
        state = torch.tensor([get_morgen_fingerprint(rt1, nBits=256)], dtype=torch.float).to("cpu")
        template_mask = np.array(get_reaction_mask(rt1, templates))
        T_mask = torch.from_numpy(template_mask.astype(np.float32)).to("cpu")
        cur_temp = np.maximum(1.0 * np.exp(-0.00006 * (step + 1)), 0.1)
        template = rxn_model(state, T_mask, cur_temp)
        template_proba = template.squeeze().detach().cpu().numpy() + 1e-10
        top_rxn_idx = (template_proba * template_mask).argsort()[::-1]
        smarts = templates[top_rxn_idx[0]]
            
        print(f'step {step} - top_rxn_idx:', top_rxn_idx[0], smarts)
        rxn = AllChem.ReactionFromSmarts(smarts)
        mol1 = Chem.MolFromSmiles(rt1)
        if len(smarts.split('>')[0].split('.')) == 1:
            rt2 = None
            ps = rxn.RunReactants((mol1, ))
        else:
            tid = check_match_smarts(mol1, smarts)
            rt2_fp = reagent_model.forward(state, template)
            rt2_fp = rt2_fp.squeeze().detach().cpu().numpy()
            rt2 = get_second_reactant(tid, smarts, rt2_fp)
            if not rt2:
                print('No match reactant 2')
                break
            mol2 = Chem.MolFromSmiles(rt2)
            if tid == 0:
                ps = rxn.RunReactants((mol1, mol2))
            else:
                ps = rxn.RunReactants((mol2, mol1))
            
        uniqps = []
        for p in ps:
            try:
                Chem.SanitizeMol(p[0])
                smi = Chem.MolToSmiles(p[0])
                uniqps.append(smi)
            except:
                continue
        uniqps = list(set(uniqps))
        
        if len(uniqps) == 0:
            print('No product')
            break
        else:
            prod = uniqps[0]
            pmol = Chem.MolFromSmiles(prod)
            mw = round(ExactMolWt(pmol), 2)
            if mw > 600:
                print('molecular weight > 600')
                break
            
        one_step_reaction.append([rt1, smarts, rt2, prod])
        rt1 = prod
        step += 1
    
    reac_num += 1
    syn_plan[str(reac_num)] = one_step_reaction
    
with open('rxn_results.json', 'w') as f:
    final_dict = {'results': syn_plan}
    f.write(json.dumps(final_dict, indent=4))
