import os, shutil

def get_filename(path,filetype, imagepath):  # 输入路径、文件类型例如'.csv'
    name = []
    for root,dirs,files in os.walk(path):
        for i in files:
            if os.path.splitext(i)[1] == filetype:
                name.append(i)

    for i in range(len(name)):
        original = os.path.join(path, name[i])
        imageDir = os.path.join(imagepath, name[i])

        shutil.move(original, imageDir)



get_filename('C:\\Users\\TME-DJ\\Desktop\\train\\train', '.jpg',
             'C:\\Users\\TME-DJ\\Desktop\\train\\images')
