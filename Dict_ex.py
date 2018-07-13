#Dictionary Examples

d={}
d=dict()
d[100]='Arun'
print(d)
d[200]='kumar'
print(d)
x={300:'Vishnu',400:'Kiran kumar'}
d.update(x)
print(d)
print(d.get(400))
print(d.get(600))
print('-------------------------')
print(d)
print(d.get(800,'Not present'))
print('d-After get default',d)
print(d[400]) 
d.setdefault(100,'Raju babu')
d.setdefault(700,'charan')
print(d)
print('len:',len(d))
print(d.keys())
for x in d.keys():
    print(x)
print(d.values())
for y in d.values():
    print(y)
print(d.items())
print('Keys:','\t\t','Values:')
print('-----------------------')
for k,v in d.items():
    print(k,'\t','---->','\t',v)
d.pop(100)
print('After pop:',d)
d.popitem()
print('After popitem:',d)
del d[200]
print('After del:',d)
d1=d.copy()
print('d1:',d1)
d.clear()
print('After clear:',d)
del d
print(d)

