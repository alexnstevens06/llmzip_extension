import difflib
import io
from core import BitOutputStream

SEQUENCE_1 = "print('Hello, World!')"
SEQUENCE_2 = "Hello, World! PEACE!"

opcode_translate = {
    "equal": [0, 0],
    "replace": [0, 1],
    "delete": [1, 0],
    "insert": [1, 1]
}


def main():
    # isjunk=None (the first argument) means we don't ignore any characters as anchors.
    matcher = difflib.SequenceMatcher(None, SEQUENCE_1, SEQUENCE_2)

    ratio = matcher.ratio()
    print(f"Target Sequence: {SEQUENCE_1}")
    print(f"Actual Sequence: {SEQUENCE_2}")
    print(f"Similarity Ratio: {ratio:.4f}")

    print("\nEdit Instructions (Opcodes):")

    # define empty bitstream
    stream = io.BytesIO()
    outstream = BitOutputStream(stream)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        outstream.write_bits(opcode_translate[tag])
        if tag == "equal":

            pass
        if tag == "replace":
            pass
        if tag == "delete":
            pass
        if tag == "insert":
            pass
        a_part = SEQUENCE_1[i1:i2]
        b_part = SEQUENCE_2[j1:j2]
        print(
            f"  {tag:7} a[{i1}:{i2}] ('{a_part}') -> b[{j1}:{j2}] ('{b_part}')")
        #


if __name__ == "__main__":
    main()
