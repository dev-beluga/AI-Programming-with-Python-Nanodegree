# f=open('./some_file.txt', 'a')
# f.write("Wssup Nigga")
# f.close()
# print('./some_file.txt')

# files = []
# for i in range(200000):
#     files.append(open('./some_file2.txt', 'r'))
#     print(i)
    
# with open('./some_file2.txt', 'r') as f:
#     file_data = f.read()   
# print(file_data)

# with open('./some_file2.txt') as word:
#     print(word.read(2))
#     print(word.read(8))
#     print(word.read())
    
new_lines = []
with open('./some_file2.txt') as f:
    for line in f:
        new_lines.append(line.strip())
        
print(new_lines)