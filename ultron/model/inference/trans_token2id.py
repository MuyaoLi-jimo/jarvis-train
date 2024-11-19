data = {127929: (0, 0), 127930: (0, 1), 127931: (0, 2), 127932: (0, 3), 127933: (0, 4), 127934: (0, 5), 127935: (0, 6), 127936: (0, 7), 127937: (0, 8), 127938: (0, 9),
        127939: (1, 0), 127940: (1, 1), 127941: (1, 2),
        127942: (2, 0), 127943: (2, 1), 127944: (2, 2),
        127945: (3, 0), 127946: (3, 1), 127947: (3, 2),
        127948: (4, 0), 127949: (4, 1),
        127950: (5, 0), 127951: (5, 1),
        127952: (6, 0), 127953: (6, 1),
        127954: (7, 0), 127955: (7, 1),
        127956: (8, 0), 127957: (8, 1),
        127958: (9, 0), 127959: (9, 1), 127960: (9, 2), 127961: (9, 3), 127962: (9, 4), 127963: (9, 5), 127964: (9, 6), 127965: (9, 7), 127966: (9, 8), 127967: (9, 9), 127968: (9, 10), 127969: (9, 11), 127970: (9, 12), 127971: (9, 13), 127972: (9, 14), 127973: (9, 15), 127974: (9, 16), 127975: (9, 17), 127976: (9, 18), 127977: (9, 19), 127978: (9, 20),
        127979: (10, 0), 127980: (10, 1), 127981: (10, 2), 127982: (10, 3), 127983: (10, 4), 127984: (10, 5), 127985: (10, 6), 127986: (10, 7), 127987: (10, 8), 127988: (10, 9), 127989: (10, 10), 127990: (10, 11), 127991: (10, 12), 127992: (10, 13), 127993: (10, 14), 127994: (10, 15), 127995: (10, 16), 127996: (10, 17), 127997: (10, 18), 127998: (10, 19), 127999: (10, 20)}

# 对每个键增加256，并保持原始值不变
updated_data = {key + 256: value for key, value in data.items()}

# 显示部分更新后的数据
print({k: updated_data[k] for k in list(updated_data)})  # 只展示更新后的前10个元素

exit()
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