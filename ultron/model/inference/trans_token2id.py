special_tokens = [
    (('<|reserved_special_token_200|>', 128205),('<|reserved_special_token_201|>', 128206),('<|reserved_special_token_202|>', 128207),('<|reserved_special_token_203|>', 128208),('<|reserved_special_token_204|>', 128209),('<|reserved_special_token_205|>', 128210),('<|reserved_special_token_206|>', 128211),('<|reserved_special_token_207|>', 128212),('<|reserved_special_token_208|>', 128213),('<|reserved_special_token_209|>', 128214),),
    (('<|reserved_special_token_210|>', 128215),('<|reserved_special_token_211|>', 128216),('<|reserved_special_token_212|>', 128217),),
    (('<|reserved_special_token_213|>', 128218),('<|reserved_special_token_214|>', 128219),('<|reserved_special_token_215|>', 128220),),
    (('<|reserved_special_token_216|>', 128221),('<|reserved_special_token_217|>', 128222),('<|reserved_special_token_218|>', 128223),),
    (('<|reserved_special_token_219|>', 128224),('<|reserved_special_token_220|>', 128225),),
    (('<|reserved_special_token_221|>', 128226),('<|reserved_special_token_222|>', 128227),),
    (('<|reserved_special_token_223|>', 128228),('<|reserved_special_token_224|>', 128229),),
    (('<|reserved_special_token_225|>', 128230),('<|reserved_special_token_226|>', 128231),),
    (('<|reserved_special_token_227|>', 128232),('<|reserved_special_token_228|>', 128233),),
    (('<|reserved_special_token_229|>', 128234),('<|reserved_special_token_230|>', 128235),('<|reserved_special_token_231|>', 128236),('<|reserved_special_token_232|>', 128237),('<|reserved_special_token_233|>', 128238),('<|reserved_special_token_234|>', 128239),('<|reserved_special_token_235|>', 128240),('<|reserved_special_token_236|>', 128241),('<|reserved_special_token_237|>', 128242),('<|reserved_special_token_238|>', 128243),('<|reserved_special_token_239|>', 128244),),
    (('<|reserved_special_token_240|>', 128245),('<|reserved_special_token_241|>', 128246),('<|reserved_special_token_242|>', 128247),('<|reserved_special_token_243|>', 128248), ('<|reserved_special_token_244|>', 128249),('<|reserved_special_token_245|>', 128250),('<|reserved_special_token_246|>', 128251),('<|reserved_special_token_247|>', 128252),('<|reserved_special_token_248|>', 128253),('<|reserved_special_token_249|>', 128254),('<|reserved_special_token_250|>', 128255),),
]

re_tokens = {
    '<|reserved_special_token_200|>': (0, 0),'<|reserved_special_token_201|>': (0, 1),'<|reserved_special_token_202|>': (0, 2),'<|reserved_special_token_203|>': (0, 3),'<|reserved_special_token_204|>': (0, 4),'<|reserved_special_token_205|>': (0, 5),'<|reserved_special_token_206|>': (0, 6),'<|reserved_special_token_207|>': (0, 7),'<|reserved_special_token_208|>': (0, 8),'<|reserved_special_token_209|>': (0, 9),
    '<|reserved_special_token_210|>': (1, 0),'<|reserved_special_token_211|>': (1, 1),'<|reserved_special_token_212|>': (1, 2),
    '<|reserved_special_token_213|>': (2, 0),'<|reserved_special_token_214|>': (2, 1),'<|reserved_special_token_215|>': (2, 2),
    '<|reserved_special_token_216|>': (3, 0),'<|reserved_special_token_217|>': (3, 1),'<|reserved_special_token_218|>': (3, 2),
    '<|reserved_special_token_219|>': (4, 0),'<|reserved_special_token_220|>': (4, 1),
    '<|reserved_special_token_221|>': (5, 0),'<|reserved_special_token_222|>': (5, 1),
    '<|reserved_special_token_223|>': (6, 0),'<|reserved_special_token_224|>': (6, 1),
    '<|reserved_special_token_225|>': (7, 0),'<|reserved_special_token_226|>': (7, 1),
    '<|reserved_special_token_227|>': (8, 0),'<|reserved_special_token_228|>': (8, 1),
    '<|reserved_special_token_229|>': (9, 0),'<|reserved_special_token_230|>': (9, 1),'<|reserved_special_token_231|>': (9, 2),'<|reserved_special_token_232|>': (9, 3),'<|reserved_special_token_233|>': (9, 4),'<|reserved_special_token_234|>': (9, 5),'<|reserved_special_token_235|>': (9, 6),'<|reserved_special_token_236|>': (9, 7),'<|reserved_special_token_237|>': (9, 8),'<|reserved_special_token_238|>': (9, 9),'<|reserved_special_token_239|>': (9, 10),
    '<|reserved_special_token_240|>': (10, 0),'<|reserved_special_token_241|>': (10, 1),'<|reserved_special_token_242|>': (10, 2),'<|reserved_special_token_243|>': (10, 3),'<|reserved_special_token_244|>': (10, 4),'<|reserved_special_token_245|>': (10, 5),'<|reserved_special_token_246|>': (10, 6),'<|reserved_special_token_247|>': (10, 7),'<|reserved_special_token_248|>': (10, 8),'<|reserved_special_token_249|>': (10, 9),'<|reserved_special_token_250|>': (10, 10)
}

# 创建 word 到 num 的映射
word_to_num = {}
for group in special_tokens:
    for word, num in group:
        word_to_num[word] = num

# 替换 re_tokens 中的 word 为 num
replaced_tokens = {}
for word, position in re_tokens.items():
    num = word_to_num[word]
    replaced_tokens[num] = position

# 输出结果
output_format = "{" + ", ".join(f"{k}:{v}" for k, v in replaced_tokens.items()) + "}"
print(output_format)