"""List PE import dependencies without requiring Visual Studio on PATH."""
import json, pathlib, struct, sys
b = pathlib.Path(sys.argv[1]).read_bytes()
pe = struct.unpack_from('<I', b, 0x3c)[0]
assert b[pe:pe+4] == b'PE\0\0'
sections = struct.unpack_from('<H', b, pe+6)[0]
optional_size = struct.unpack_from('<H', b, pe+20)[0]
optional = pe+24
directory = optional+(112 if struct.unpack_from('<H',b,optional)[0]==0x20b else 96)
section_start = optional+optional_size
def offset(rva):
    for i in range(sections):
        s = section_start+i*40
        size,base,rawsize,raw = struct.unpack_from('<IIII',b,s+8)
        if base <= rva < base+max(size,rawsize): return raw+rva-base
    return rva
def string(rva):
    start=offset(rva)
    return b[start:b.index(b'\0',start)].decode()
imports=[]
rva=struct.unpack_from('<I',b,directory+8)[0]
if rva:
    i=offset(rva)
    while any(b[i:i+20]):
        imports.append(string(struct.unpack_from('<I',b,i+12)[0])); i+=20
print(json.dumps(dict(binary=sys.argv[1],bytes=len(b),imports=sorted(imports)),indent=2))
