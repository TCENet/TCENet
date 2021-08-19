ioc_type_mapping_dict = {
    "ipv4addr": "ip",
    "ipv6addr": "ip",
    "ipv4range": "ip",
    "ipv6range": "ip",
    "ipv4cidr": "ip",
    "ipv6cidr": "ip",
    "fqdn": "fqdn",
    "email": "email",
    "filename": "filename",
    "url": "url",
    "md5": "hash",
    "sha1": "hash",
    "sha256": "hash",
    "filepath": "filepath",
    "regkey": "regkey",
    "cve": "cve"
}

BTR_ioc_type_dict = {
    "ip": 0,
    "fqdn": 1,
    "email": 2,
    "filename": 3,
    "url": 4,
    "hash": 5,
    "filepath": 6,
    "regkey": 7,
    "cve": 8,
    "codemethod": 9,
    "protocol": 10,
    "dataobject": 11
}



encode_decode_method_list = [" aes ", " aes", "aes-", " xor", "xor-", " ror", " base64", " rc4", " des ", " des-",
                             " lznt1", " cast-", " 3des", " lzo"]
protocol = ["http", "https", "http/https", "ftp", "smtp", "pop3", "dns"]

collection_data_object = ["desktop", "clipboard", "directory", "exchange", "gmail", "outlook", "mailbox", "keystroke",
                          "keylogger", "password"]

Technique_key_verb = {
    "T1027.txt": [
        "obfuscate",
        "encrypt",
        "encode",
        "include",
        "compress"
    ],
    "TA0009": [
        "capture",
        "steal",
        "collect",
        "watch"
    ],
    "T1071": [
        "communicate",
        "compromise",
        "tunnel"
    ],
    "T1140": [
        "decode",
        "decrypt",
        "encode",
        "store"
    ],
    "T1053.005": [
        "schedule",
        "establish",
        "create",
        "execute",
        "run",
        "launch"
    ],
    "T1566.txt": [
        "send",
        "phish",
        "contain",
        "deliver",
        "attach",
        "target",
        "compromise"
    ]
}
