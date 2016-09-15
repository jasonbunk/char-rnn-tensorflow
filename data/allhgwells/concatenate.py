import os, re, sys

# settings
lowercase = False
searchpath = 'processed'

# regex pattern
pattern = re.compile(r'[a-zA-Z \.,\'!?;\-\"\n]')

# get files in preprocessed folder
filesinfold = [f for f in os.listdir(searchpath) if f.endswith('.txt')]

# start concatenating
with open('input.txt', 'w') as outconcat:
	for fname in filesinfold:
		nlines = 0
		with open(os.path.join(searchpath,fname), 'r') as infile:
			for line in infile:
				# end-of-lines should just be \n, not \r\n
				line = line.replace('\r','')

				# check for invalid characters, warn user
				outofvocab = re.sub(pattern,'',line)
				if len(outofvocab) > 0:
					print("file \'"+fname+"\' contained \'"+str(outofvocab)+"\' out-of-vocab characters")

				# strip invalid characters and make lowercase
				line = ''.join(re.findall(pattern,line))
				if lowercase:
					line = line.lower()

				outconcat.write(line)
				nlines += 1


