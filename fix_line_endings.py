# This script converts CRLF to LF in the specified file
input_path = 'sam2/checkpoints/download_ckpts.sh'
with open(input_path, 'rb') as f:
    content = f.read()
content = content.replace(b'\r\n', b'\n')
with open(input_path, 'wb') as f:
    f.write(content)
print('Line endings fixed for', input_path) 