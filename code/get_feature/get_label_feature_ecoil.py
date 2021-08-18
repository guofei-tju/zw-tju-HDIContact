import pickle
import numpy
import sys
import os
import matplotlib as mpl
import warnings
import xlrd
from Bio import PDB
import Bio.PDB
from Bio.PDB.PDBParser import PDBParser
from itertools import combinations_with_replacement
from scipy.sparse import  csr_matrix
warnings.filterwarnings("ignore")
mpl.use('Agg')
def get_coord_dis(atom1,atom2):
    dis_atom = atom1-atom2
    return numpy.sqrt(numpy.square(dis_atom[0])+numpy.square(dis_atom[1])+numpy.square(dis_atom[2]))
def get_residues(pdb_fn, model_num=0):
    """Build a simple list of residues from a single chain of a PDB file.
    Args:
        pdb_fn: The path to a PDB file.
        chain_ids: A list of single-character chain identifiers.
        model_num: The model number in the PDB file to use (optional)
    Returns:
        A list of Bio.PDB.Residue objects.
    """
    pdb_id = os.path.splitext(os.path.basename(pdb_fn))[0]
    parser = Bio.PDB.PDBParser(pdb_id, pdb_fn)
    struct = parser.get_structure(pdb_id, pdb_fn)
    model = struct[model_num]
    chains = model.get_list()
    residues = []
    for ch in chains:
        for res in ch.get_residues():
            residues.append(res)
    return residues
def calc_dist_matrix(residues):
    mat = numpy.zeros((len(residues), len(residues)), dtype="float64")

    # 任意两个元素的所有排列组合结果，忽略成对中元素顺序
    pair_indices = combinations_with_replacement(range(len(residues)), 2)

    for i, j in pair_indices:
        if i != j :
            res_a = residues[i]
            res_b = residues[j]
            res_min_dis = []
            for atom1 in res_a:
                for atom2 in res_b:
                    res_min_dis.append(get_coord_dis(atom1.get_coord(), atom2.get_coord()))
            mat[i, j] = min(res_min_dis)
            mat[j,i] = min(res_min_dis)
        else:
            mat[i,j] = 100
            mat[j,i] = 100
    return mat
def get_atom_coord(res, atom_name):

    try:
        coord = res[atom_name].get_coord()
    except KeyError:
        if atom_name != "CB":
            # Return the first/only available atom.
            atom = res.child_dict.values()[0]
            sys.stderr.write(
                "WARNING: {} atom not found in {}".format(atom_name, res)
                    + os.linesep
            )
            return atom.get_coord()
        if "N" in res and "CA" not in res:
            coord = res["N"].get_coord()
        else:
            assert ("N" in res)
            assert ("CA" in res)
            assert ("C" in res)

            # Infer the CB atom position described in http://goo.gl/OaNjxe
            #
            # NOTE:
            # These are Bio.PDB.Vector objects and _not_ numpy arrays.
            N = res["N"].get_rector()
            CA = res["CA"].get_rector()
            C = res["C"].get_rector()

            CA_N = N - CA
            CA_C = C - CA

            rot_mat = Bio.PDB.rotaxis(-numpy.pi * 120.0 / 180.0, CA_C)

            coord = (CA + CA_N.left_multiply(rot_mat)).get_array()
    return coord
def calc_dist_matrix_cb(residues, measure="CB"):
    """Calculate the distance matrix for a list of residues.
    Args: residues: A list of ``Bio.PDB.Residue`` objects.
        measure: The distance measure (optional).
    Returns: The distance matrix as a masked array.
    """
    mat = numpy.zeros((len(residues), len(residues)), dtype="float64")

    mat[:] = numpy.nan

    # 任意两个元素的所有排列组合结果，忽略成对中元素顺序
    pair_indices = combinations_with_replacement(range(len(residues)), 2)

    for i, j in pair_indices:
        res_a = residues[i]
        res_b = residues[j]
        A = get_atom_coord(res_a, measure)
        B = get_atom_coord(res_b, measure)
        dist = numpy.linalg.norm(A - B)
        mat[i,j] = dist
        mat[j,i] = dist

    return mat

if __name__ == '__main__':
    #address configure

    PDB_DIR = "F:/complex_pdb/EV-Ecoil/PDB_structure/"
    Information_path = 'F:/complex_pdb/EV-Ecoil/Ecoil_list.xlsx'
    save_directory = "F:/paper2exper/data/Ecoil_testdataset_distance.cpkl"
    a2m_add = "F:/complex_pdb/EV-Ecoil/ev_ecoil/"
    workbook = xlrd.open_workbook(Information_path)
    sheet = workbook.sheet_by_index(4)

    protein_infor_list = []

    for i in range(1, sheet.nrows):

        pdb_id = sheet.cell(i, 0).value
        chain_L = str(sheet.cell(i, 1).value)
        chain_R = str(sheet.cell(i, 2).value)
        name = pdb_id+'_'+chain_L+chain_R+sheet.cell(i, 11).value
        print(">"+name)
        residues = []
        pdb_fn_l = os.path.join(PDB_DIR, name + "_l.pdb")
        pdb_fn_r = os.path.join(PDB_DIR, name + "_r.pdb")
        parser = PDB.PDBParser()
        name_l = pdb_id + "_l"
        structure = parser.get_structure(name_l, pdb_fn_l)
        ppb = PDB.PPBuilder()

        seq_l = ""
        for pp in ppb.build_peptides(structure[0]):
            seq_l = seq_l + pp.get_sequence()
            for ress in pp:
                residues.append(ress)
        name_r = pdb_id + "_r"
        structure = parser.get_structure(name_r, pdb_fn_r)
        ppb = PDB.PPBuilder()

        seq_r = ""
        for pp in ppb.build_peptides(structure[0]):
            seq_r = seq_r + pp.get_sequence()
            for ress in pp:
                residues.append(ress)
        '''
        a2m_file =a2m_add+pdb_id+chain_L+chain_R+'_'+sheet.cell(i, 6).value+'.a2m'
        with open(a2m_file, 'r') as f:
            list = f.readlines()
        seq = ""
        for context in list[1:]:
            if context[0]==">":
                break
            seq = seq+context[:-1]

        print(seq)
        '''
        res = residues
        distance = calc_dist_matrix(res)
        #Label = numpy.where(distance <= 10, 1, 0)  #X += X.T - np.diag(X.diagonal())
        #distance = calc_dist_matrix_cb(res, measure="CB")
        #Label = numpy.where(distance <= 8, 1, 0)
        # distance = calc_dist_matrix_cb(res, measure="CA")
        # Label = numpy.where(distance <= 8, 1, 0)
        protein = {"Complex_code": name,
                   "msa_name":pdb_id+chain_L+chain_R+'_'+sheet.cell(i, 6).value,
                   "r_seq": seq_r, "l_seq": seq_l,
                   "r_length": len(seq_r), "l_length": len(seq_l),
                   "Distance_Label": distance}
        protein_infor_list.append(protein)
        # print(">"+name_l[:4],len_l,len_r)
        # print(seq_r)
        '''
        if numpy.sum(numpy.sum(Label[:len(seq_l), len(seq_l):len(seq_l) + len(seq_r)] == 1)) > 0:
            protein_infor_list.append(protein)
        else:
            print(name + "  map zero !!!")
        print(i, name + " finish ...")
        '''
        # save
    f = open(save_directory, 'wb')
    pickle.dump(protein_infor_list, f)
    f.close()




