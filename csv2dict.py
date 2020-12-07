import pandas as pd


path = "C:\\Users\\TME\\Desktop\\dataset\\train\\_annotations.csv"

df = pd.read_csv(path, index_col=0)[['xmin','ymin','xmax', 'ymax']]



df['id'] = 0.0
df['xmin'] = df.apply(lambda x: x['xmin']/416, axis=1)
df['ymin'] = df.apply(lambda x: x['ymin']/416, axis=1)
df['xmax'] = df.apply(lambda x: x['xmax']/416, axis=1)
df['ymax'] = df.apply(lambda x: x['ymax']/416, axis=1)
df2 =df.T
df3 = df2.to_dict('list')
f = open("C:\\Users\\TME\\Desktop\\dataset\\annotation.index", 'w')
f.write(str(df3))
f.close()
