import bitstring


PMF = [0.5, 0.5]
string = "abba"
decode = {"a": 0, "b": 1}
encode = {v: k for k, v in decode.items()}

bt = bitstring.BitArray(hex="123")
bt2 = bitstring.BitArray(hex="abc")

print(bt.bin)
print(bt2.bin)
print((bt+bt2).bin)

bt.append(bt2)
print(bt.bin)
