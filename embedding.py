#!/home/zengxiangxiang/hyy/miniconda3/bin/python3.6
import subprocess
def main():
    command = 'deepwalk  --outpu  save/blogcatalog/blogcatalog(fake).embeddings --extra save/blogcatalog/blogcatalog_final.txt'
    subprocess.call(command,shell=True)
if __name__=='__main__':
     main()
