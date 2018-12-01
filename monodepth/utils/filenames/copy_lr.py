

with open(fname) as f:
    Llines = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
Llines = [x.strip() for x in Llines] 
Rlines = [x.replace('left', 'right') for x in Llines]
combLines = [l+' '+r for l,r in zip(Llines, Rlines)]
with open(combfname, 'w') as f:
	for item in combLines:
		f.write('%s\n' %item)




