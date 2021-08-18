#准备标签和特征，特征为L+R * L+R *660的attention map
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
import matplotlib.pyplot as plt
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
            N = res["N"].get_vector()
            CA = res["CA"].get_vector()
            C = res["C"].get_vector()

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

    PDB_DIR = "F:/complex_pdb/bound_structures/"
    save_directory = "F:/ubuntu_file/data/Baker_dataset/baker31_dataset_min10.cpkl"
    Information_path = 'F:/ubuntu_file/data/Baker_dataset/baker_list.xlsx'
    a2m_add = "F:/ubuntu_file/data/Baker_dataset/msas_baker/"
    workbook = xlrd.open_workbook(Information_path)
    sheet = workbook.sheet_by_index(0)
    protein_infor_list = []
    for i in range(1, sheet.nrows):
        pdb_id = sheet.cell(i, 0).value

        pdb_fn_r = os.path.join(PDB_DIR, "{}_r_b.pdb".format(pdb_id))
        pdb_fn_l = os.path.join(PDB_DIR, "{}_l_b.pdb".format(pdb_id))
        parser = PDB.PDBParser()
        residues = []

        name = pdb_id + "_l"
        structure = parser.get_structure(name, pdb_fn_l)
        ppb = PDB.PPBuilder()

        seq = ""
        for pp in ppb.build_peptides(structure[0]):
            t = pp
            seq = seq + pp.get_sequence()
            for ress in pp:
                residues.append(ress)
        len_l = len(seq)
        seq_l = seq
        name_l = name


        name = pdb_id+"_r"
        structure = parser.get_structure(name, pdb_fn_r)
        ppb = PDB.PPBuilder()

        seq=""
        for pp in ppb.build_peptides(structure[0]):
            seq = seq+pp.get_sequence()
            for ress in pp:
                residues.append(ress)

        len_r = len(seq)
        seq_r = seq
        name_r = name



        if len_l + len_r <=1023:

            # complex dis map->figure label
            res = residues
            distance = calc_dist_matrix(res)
            Label = numpy.where(distance <= 10, 1, 0)  #X += X.T - np.diag(X.diagonal())
            #distance = calc_dist_matrix_cb(res, measure="CB")
            #Label = numpy.where(distance <= 8, 1, 0)
            #distance = calc_dist_matrix_cb(res, measure="CA")
            #Label = numpy.where(distance <= 12, 1, 0)
            protein = {"Complex_code":name_l[:4],
                "Ligand_seq": seq_l, "Receptor_seq": seq_r,
                "Ligand_length": len_l, "Receptor_length": len_r,
                       "Map_Label":Label}
            #print(">"+name_l[:4],len_l,len_r)
            #print(seq_r)

            if numpy.sum(numpy.sum(Label[:len_l,len_l:len_l+len_r]==1))>0:
                protein_infor_list.append(protein)


                a2m_file =a2m_add+pdb_id+'.a3m'
                with open(a2m_file, 'r') as f:
                    list = f.readlines()
                seq_msa = ""
                for context in list[1:]:
                    if context[0]==">":
                        break
                    seq_msa = seq_msa+context[:-1]
                print(">"+pdb_id)
                print(seq_l+seq_r)
                print(seq_msa)

            else:
                print(name_l[:4] + "  map zero !!!")
            print(i,name_l[:4]+" finish ...")

    #save
    f = open(save_directory, 'wb')
    pickle.dump(protein_infor_list, f)
    f.close()
