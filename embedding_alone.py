import subprocess
def main():
    #command = 'deepwalk --input data/GrQc/GrQc.edgelist --outpu  save/GrQc/GrQc.embeddings --extra save/GrQc/GrQc_final.txt'
   #command = 'deepwalk  --outpu  save/GrQc/GrQc_test(fake).embeddings --extra save/GrQc/GrQc_final.txt'
    command = 'deepwalk  --outpu  save/PPI/HI-II-14.embeddings --extra save/PPI/HI-II-14_final.txt'
   # command = 'deepwalk  --input data/blogcatalog/blogcatalog.edgelist --outpu save/blogcatalog/blogcatalog(deepwalk).embeddings'
    subprocess.call(command, shell=True)
if __name__ == '__main__':
    main()
