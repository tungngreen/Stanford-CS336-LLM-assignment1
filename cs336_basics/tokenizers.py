def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

def decode_utf8_bytes_to_str(bytestring: bytes):
    return bytestring.decode("utf-8")

if __name__ == "__main__":
    print(decode_utf8_bytes_to_str("삶".encode("utf-8"))) 