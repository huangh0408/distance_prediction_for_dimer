from __future__ import division
import numpy as np
from scipy.misc import imsave
import os,sys
#from __future__ import division

def plot(a):
	pdb_name_file="./workspace/pdb_list.txt"
	f=open(pdb_name_file,"r")
	for line in f:
		temp=line.split()
		pdb_name=temp[0]
#		chain_file_name=os.path.join('./workspace/',pdb_name+'_temp_dir/',pdb_name+'_chain.txt')
		chain_file_name=os.path.join('./chain/',pdb_name+'_chain.txt')
		ff=open(chain_file_name,"r")
		for rr in ff:
		#	for ll in ff:			
			temp_1=rr.split()
#			temp_2=ll.split()
			r=temp_1[0]
			l=temp_1[1]
			#if r < l:
			matrix_A_name=os.path.join('./true_contact_matrix/',pdb_name+'_'+r+'_temp'+'.contact_matrix')
			matrix_B_name=os.path.join('./true_contact_matrix/',pdb_name+'_'+l+'_temp'+'.contact_matrix')
			matrix_A_B_name=os.path.join('./true_contact_matrix/',pdb_name+'_'+r+'_'+l+'_temp'+'.contact_matrix')
#			matrix_B_A_name=os.path.join('./true_contact_matrix/',pdb_name+'_'+l+'_'+r+'_temp'+'.contact_matrix')
			exist_A=os.path.exists(matrix_A_name)
			exist_B=os.path.exists(matrix_B_name)
			exist_A_B=os.path.exists(matrix_A_B_name)
			if exist_A==True and exist_B==True and exist_A_B==True:
				size_A=os.path.getsize(matrix_A_name)
				size_B=os.path.getsize(matrix_B_name)
				size_A_B=os.path.getsize(matrix_A_B_name)
				print (pdb_name,r,l)
#			size_B_A=os.path.getsize(matrix_B_A_name)
				if size_A > 0 and size_B > 0 and size_A_B > 0:
					matrix_A=np.loadtxt(matrix_A_name)
					matrix_B=np.loadtxt(matrix_B_name)
					matrix_A_B=np.loadtxt(matrix_A_B_name)
					sum_A_B=matrix_A_B.sum()
#					matrix_B_A=np.transpose(matrix_A_B)
					l_A=len(matrix_A)
					l_B=len(matrix_B)
					matrix_B_A=np.zeros((l_B,l_A))
					l_A_1=matrix_A_B.shape[0]
					l_B_1=matrix_A_B.shape[1]
					ratio_A_B=sum_A_B/(l_A+l_B)
					if l_A == l_A_1 and l_B == l_B_1 and ratio_A_B > 0.1 and l_A>50 and l_B>50:
						M1=np.hstack((matrix_A,matrix_A_B))
						M2=np.hstack((matrix_B_A,matrix_B))
						flag_A_B=np.vstack((M1,M2))
						flag_A_B_matrix_name=os.path.join('./flag_contact_matrix/',pdb_name+'_'+r+'_'+l+'.txt')
						flag_A_B_image_name=os.path.join('./flag_contact_image/',pdb_name+'_'+r+'_'+l+'.jpg')
						flag_A_B_length_name=os.path.join('./flag_contact_length/',pdb_name+'_'+r+'_'+l+'.txt')
						f=open(flag_A_B_length_name,"w")
						print >> f, "%d %d" % (l_A, l_B)
						imsave(flag_A_B_image_name,flag_A_B)
						np.savetxt(flag_A_B_matrix_name,flag_A_B)
		


if __name__=="__main__":
	
	a=1
	plot(a)

