import re
import numpy as np
from collections import OrderedDict
from jarvis.arm.utils.vpt_lib.actions import  ActionTransformer,Buttons
from jarvis.arm.utils.vpt_lib.action_mapping import CameraHierarchicalMapping


def get_special_token(model_id:str = '/nfs-shared/models/llama-3', bases:list = [10,3,3,3,2,2,2,2,2,11,11]) -> list:  #假设永远不会出现8641这个数
    '''
    bases: button+camera
    :output: list, 返回一个包含所有未知token的列表
    Function: 生成一个包含所有未知token的列表, 用于标记未知的token
    Examples:
    '''
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    token_num = sum(bases)+10
    special_tokens = sorted(list(tokenizer.vocab.items()), key=lambda x: x[-1])[-token_num:]
    return special_tokens

'''
Llama-2, Vicuna-v1.5 
[
    ('진', 31536),('জ', 31537),('천', 31563),('년', 31571),('세', 31578),('민', 31582),('ർ', 31585),('ἡ', 31598),('호', 31603),('ਰ', 31604),
    ('그', 31607),('න', 31609),('ན', 31614),
    ('ゆ', 31621),('ご', 31622),('현', 31680), 
    ('군', 31699), ('무', 31716), ('위', 31724), 
    ('안', 31734), ('박', 31736),
    ('용', 31737), ('단', 31746), 
    ('면', 31747), ('남', 31754), 
    ('강', 31774), ('씨', 31781), 
    ('개', 31789), ('들', 31804), 
    ('차', 31817), ('학', 31822), ('만', 31826), ('터', 31856), ('식', 31895), ('과', 31906), ('타', 31925), ('종', 31930), ('내', 31940), ('중', 31941), ('방', 31945), 
    ('월', 31950), ('회', 31953), ('모', 31962), ('바', 31963), ('음', 31966), ('재', 31973), ('명', 31976), ('합', 31980), ('역', 31987), ('백', 31989), ('왕', 31996), 
]
llava-v1.6 (llava-v1.6-mistral-7b-hf,llava-v1.6-vicuna-13b-hf,llava-v1.6-vicuna-7b-hf )
[
    ('朱', 31947),('ǝ', 31948),('Ḩ', 31949),('担', 31950),('灰', 31951), ('讲', 31952), ('롤', 31953),('😤', 31955),('ោ', 31956),('애', 31957),
    ('였', 31958),('질', 31959),('振', 31960),
    ('灯', 31961),('ĉ', 31962),('ස', 31963),
    ('閉', 31964),('램', 31965),('ಂ', 31966),
    ('げ', 31967),('̧', 31968),
    ('狂', 31969),('融', 31970),
    ('仍', 31971),('實', 31972),
    ('楽', 31973),('範', 31974),
    ('వ', 31976),('嵌', 31977),
    ('摩', 31978),('袁', 31979),('ষ', 31980),('乎', 31981),('규', 31982),('岗', 31983),('糊', 31984),('క', 31985),('雲', 31986),('심', 31987),('ई', 31988),
    ('འ', 31989),('ἡ', 31990),('丝', 31991),('Ħ', 31992),('ٍ', 31993),('ٓ', 31994),('အ', 31995),('執', 31996),('벨', 31997),('ゼ', 31998),('梦', 31999),
]
llama-3,llava-next(llama3-llava-next-8b-hf)
[
    ('<|reserved_special_token_200|>', 128205),('<|reserved_special_token_201|>', 128206),('<|reserved_special_token_202|>', 128207),('<|reserved_special_token_203|>', 128208),('<|reserved_special_token_204|>', 128209),('<|reserved_special_token_205|>', 128210),('<|reserved_special_token_206|>', 128211),('<|reserved_special_token_207|>', 128212),('<|reserved_special_token_208|>', 128213),('<|reserved_special_token_209|>', 128214),
    ('<|reserved_special_token_210|>', 128215),('<|reserved_special_token_211|>', 128216),('<|reserved_special_token_212|>', 128217),
    ('<|reserved_special_token_213|>', 128218),('<|reserved_special_token_214|>', 128219),('<|reserved_special_token_215|>', 128220),
    ('<|reserved_special_token_216|>', 128221),('<|reserved_special_token_217|>', 128222),('<|reserved_special_token_218|>', 128223),
    ('<|reserved_special_token_219|>', 128224),('<|reserved_special_token_220|>', 128225),
    ('<|reserved_special_token_221|>', 128226),('<|reserved_special_token_222|>', 128227),
    ('<|reserved_special_token_223|>', 128228),('<|reserved_special_token_224|>', 128229),
    ('<|reserved_special_token_225|>', 128230),('<|reserved_special_token_226|>', 128231),
    ('<|reserved_special_token_227|>', 128232),('<|reserved_special_token_228|>', 128233),
    ('<|reserved_special_token_229|>', 128234),('<|reserved_special_token_230|>', 128235),('<|reserved_special_token_231|>', 128236),('<|reserved_special_token_232|>', 128237),('<|reserved_special_token_233|>', 128238),('<|reserved_special_token_234|>', 128239),('<|reserved_special_token_235|>', 128240),('<|reserved_special_token_236|>', 128241),('<|reserved_special_token_237|>', 128242),('<|reserved_special_token_238|>', 128243),('<|reserved_special_token_239|>', 128244),
    ('<|reserved_special_token_240|>', 128245),('<|reserved_special_token_241|>', 128246),('<|reserved_special_token_242|>', 128247),('<|reserved_special_token_243|>', 128248), ('<|reserved_special_token_244|>', 128249),('<|reserved_special_token_245|>', 128250),('<|reserved_special_token_246|>', 128251),('<|reserved_special_token_247|>', 128252),('<|reserved_special_token_248|>', 128253),('<|reserved_special_token_249|>', 128254),('<|reserved_special_token_250|>', 128255),
]
Fuyu
[
    ('s▁flu', 262109), ('all▁our', 262110), ('og▁i', 262111), ('t▁tri', 262112), ('ank▁you', 262113),('nia▁i', 262114), ('le▁tr', 262115), ('s▁doing▁the', 262116),
    ('makes▁the', 262117), ('createT', 262118), ('refectur', 262119), ('aditional', 262120), ('staken', 262121), ('▁a▁Mar', 262122), ('▁the▁Dis', 262123), ('▁other▁dev', 262124), 
    ('servato', 262125), ('others▁who', 262126), ('de▁pos', 262127), ('esirable', 262128), ('sign▁an', 262129), ('age▁rating', 262130), ('lementation', 262131), ('wondering▁what', 262132),
    ('st▁at▁the', 262133), ('which▁is▁an', 262134), ('in▁light▁of▁the', 262135), ('plied▁by', 262136),('se▁det', 262137), ('avies', 262138),
    ('of▁pro', 262139), ('▁the▁pheno', 262140), ('This▁study▁was', 262141), ('ion▁temperature', 262142), ('cause▁everyone', 262143)
]
'''

def map_control_token(num:int, place:int, tokenizer_type:str = "llama-2",not_text=False) -> str:
    if tokenizer_type == "llama-2":
        special_tokens = [
            (('진', 31536),('জ', 31537),('천', 31563),('년', 31571),('세', 31578),('민', 31582),('ർ', 31585),('ἡ', 31598),('호', 31603),('ਰ', 31604),),
            (('그', 31607),('න', 31609),('ན', 31614),),
            (('ゆ', 31621),('ご', 31622),('현', 31680),),
            (('군', 31699), ('무', 31716), ('위', 31724),),
            (('안', 31734), ('박', 31736),),
            (('용', 31737), ('단', 31746),),
            (('면', 31747), ('남', 31754),),
            (('강', 31774), ('씨', 31781),),
            #(('개', 31789), ('들', 31804),),
            (('차', 31817), ('학', 31822), ('만', 31826), ('터', 31856), ('식', 31895), ('과', 31906), ('타', 31925), ('종', 31930), ('내', 31940), ('중', 31941), ('방', 31945)),
            (('월', 31950), ('회', 31953), ('모', 31962), ('바', 31963), ('음', 31966), ('재', 31973), ('명', 31976), ('합', 31980), ('역', 31987), ('백', 31989), ('왕', 31996)),
        ]
    elif tokenizer_type == "llava-v1.6":
        special_tokens = [
            (('朱', 31947),('ǝ', 31948),('Ḩ', 31949),('担', 31950),('灰', 31951), ('讲', 31952), ('롤', 31953),('😤', 31955),('ោ', 31956),('애', 31957),),
            (('였', 31958),('질', 31959),('振', 31960),),
            (('灯', 31961),('ĉ', 31962),('ස', 31963),),
            (('閉', 31964),('램', 31965),('ಂ', 31966),),
            (('げ', 31967),('ふ', 31896),),
            (('狂', 31969),('融', 31970),),
            (('仍', 31971),('實', 31972),),
            (('楽', 31973),('範', 31974),),
            #(('వ', 31976),('嵌', 31977),),
            (('摩', 31978),('袁', 31979),('ষ', 31980),('乎', 31981),('규', 31982),('岗', 31983),('糊', 31984),('క', 31985),('雲', 31986),('심', 31987),('ई', 31988),),
            (('འ', 31989),('ἡ', 31990),('丝', 31991),('Ħ', 31992),('伝', 31993),('컨', 31885),('အ', 31995),('執', 31996),('벨', 31997),('ゼ', 31998),('梦', 31999),),
        ]
    elif tokenizer_type == "llama-3":
        special_tokens = [
            (('<|reserved_special_token_200|>', 128205),('<|reserved_special_token_201|>', 128206),('<|reserved_special_token_202|>', 128207),('<|reserved_special_token_203|>', 128208),('<|reserved_special_token_204|>', 128209),('<|reserved_special_token_205|>', 128210),('<|reserved_special_token_206|>', 128211),('<|reserved_special_token_207|>', 128212),('<|reserved_special_token_208|>', 128213),('<|reserved_special_token_209|>', 128214),),
            (('<|reserved_special_token_210|>', 128215),('<|reserved_special_token_211|>', 128216),('<|reserved_special_token_212|>', 128217),),
            (('<|reserved_special_token_213|>', 128218),('<|reserved_special_token_214|>', 128219),('<|reserved_special_token_215|>', 128220),),
            (('<|reserved_special_token_216|>', 128221),('<|reserved_special_token_217|>', 128222),('<|reserved_special_token_218|>', 128223),),
            (('<|reserved_special_token_219|>', 128224),('<|reserved_special_token_220|>', 128225),),
            (('<|reserved_special_token_221|>', 128226),('<|reserved_special_token_222|>', 128227),),
            (('<|reserved_special_token_223|>', 128228),('<|reserved_special_token_224|>', 128229),),
            (('<|reserved_special_token_225|>', 128230),('<|reserved_special_token_226|>', 128231),),
            #(('<|reserved_special_token_227|>', 128232),('<|reserved_special_token_228|>', 128233),),
            (('<|reserved_special_token_229|>', 128234),('<|reserved_special_token_230|>', 128235),('<|reserved_special_token_231|>', 128236),('<|reserved_special_token_232|>', 128237),('<|reserved_special_token_233|>', 128238),('<|reserved_special_token_234|>', 128239),('<|reserved_special_token_235|>', 128240),('<|reserved_special_token_236|>', 128241),('<|reserved_special_token_237|>', 128242),('<|reserved_special_token_238|>', 128243),('<|reserved_special_token_239|>', 128244),),
            (('<|reserved_special_token_240|>', 128245),('<|reserved_special_token_241|>', 128246),('<|reserved_special_token_242|>', 128247),('<|reserved_special_token_243|>', 128248), ('<|reserved_special_token_244|>', 128249),('<|reserved_special_token_245|>', 128250),('<|reserved_special_token_246|>', 128251),('<|reserved_special_token_247|>', 128252),('<|reserved_special_token_248|>', 128253),('<|reserved_special_token_249|>', 128254),('<|reserved_special_token_250|>', 128255),),
        ]
    else:
        raise ValueError(f"The tokenizer type {tokenizer_type} is not supported in control tokens.")
    return special_tokens[place][num][not_text]

def prepare_for_remap_control_token(tokenizer_type:str = "llama-2",bases:list = [10,3,3,3,2,2,2,2,2,11,11]):
    
    tokens = {}
    for i,base in enumerate(bases):
        for j in range(base):
            token = map_control_token(j,i,tokenizer_type)
            tokens[token]=(i,j)
    return tokens

def remap_control_token(token:str, tokenizer_type:str = "llama-2")->tuple:
    """由token映射到action，注意，虽然把camera从token中去掉，但是还需要它 """
    re_tokens = {}
    if tokenizer_type == "llama-2":
        re_tokens = {
            '진': (0, 0),'জ': (0, 1),'천': (0, 2),'년': (0, 3),'세': (0, 4),'민': (0, 5),'ർ': (0, 6),'ἡ': (0, 7),'호': (0, 8),'ਰ': (0, 9),
            '그': (1, 0),'න': (1, 1),'ན': (1, 2),
            'ゆ': (2, 0),'ご': (2, 1),'현': (2, 2),
            '군': (3, 0),'무': (3, 1),'위': (3, 2),
            '안': (4, 0),'박': (4, 1),
            '용': (5, 0),'단': (5, 1),
            '면': (6, 0),'남': (6, 1),
            '강': (7, 0),'씨': (7, 1),
            '개': (8, 0),'들': (8, 1),
            '차': (9, 0),'학': (9, 1),'만': (9, 2),'터': (9, 3),'식': (9, 4),'과': (9, 5),'타': (9, 6),'종': (9, 7),'내': (9, 8),'중': (9, 9),'방': (9, 10),
            '월': (10, 0),'회': (10, 1),'모': (10, 2),'바': (10, 3),'음': (10, 4),'재': (10, 5),'명': (10, 6),'합': (10, 7),'역': (10, 8),'백': (10, 9),'왕': (10, 10)
        }
    elif tokenizer_type=="llava-v1.6":
        re_tokens = {
            '朱': (0, 0),'ǝ': (0, 1),'Ḩ': (0, 2),'担': (0, 3),'灰': (0, 4),'讲': (0, 5),'롤': (0, 6),'😤': (0, 7),'ោ': (0, 8),'애': (0, 9),
            '였': (1, 0),'질': (1, 1),'振': (1, 2),
            '灯': (2, 0),'ĉ': (2, 1),'ස': (2, 2),
            '閉': (3, 0),'램': (3, 1),'ಂ': (3, 2),
            'げ': (4, 0),'ふ': (4, 1),
            '狂': (5, 0),'融': (5, 1),
            '仍': (6, 0),'實': (6, 1),
            '楽': (7, 0),'範': (7, 1),
            'వ': (8, 0),'嵌': (8, 1),
            '摩': (9, 0),'袁': (9, 1),'ষ': (9, 2),'乎': (9, 3),'규': (9, 4),'岗': (9, 5),'糊': (9, 6),'క': (9, 7),'雲': (9, 8),'심': (9, 9),'ई': (9, 10),
            'འ': (10, 0),'ἡ': (10, 1),'丝': (10, 2),'Ħ': (10, 3),'伝': (10, 4),'컨': (10, 5),'အ': (10, 6),'執': (10, 7),'벨': (10, 8),'ゼ': (10, 9),'梦': (10, 10)
        }
    elif tokenizer_type=="llama-3":
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
    else:
        raise ValueError(f"The tokenizer type {tokenizer_type} is not supported in control tokens.")
    return re_tokens.get(token,(-1,-1))


def tag_token(place, tokenizer_type:str = "llama-2"):
    """引入头标记和尾标记 """
    assert place in {0,1}
    if tokenizer_type == "llama-2":
        special_tokens = [('유', 31533),('요', 31527)]
    elif tokenizer_type == "llava-v1.6":
        special_tokens = [('ಮ', 31941),('አ', 31942)]
    elif tokenizer_type=="llama-3":
        special_tokens = [('<|reserved_special_token_199|>', 128204),('<|reserved_special_token_198|>', 128203)]
    else:
        raise ValueError(f"The tokenizer type {tokenizer_type} is not supported in control tokens.")
    return special_tokens[place][0]



def token_2_action(tokens:str, tokenizer_type:str = 'llama-2',bases:list = [10,3,3,3,2,2,2,2,2,11,11]) -> tuple:
    """将一个输入序列转换回 """
    pattern = f'{tag_token(0,tokenizer_type)}.*{tag_token(1,tokenizer_type)}'
    match = re.search(pattern, tokens)
    actions = [0]*len(bases) #初始化
    camera_null = [bases[-1]//2,bases[-2]//2]
    if not match:
        return actions

    control_tokens = match.group()[1:-1]
    for token in control_tokens:
        place,num = remap_control_token(token,tokenizer_type)
        if place!=-1:
            actions[place]=num
    # 如果移动了视野，camera button变为1
    if actions[-2:] != [bases[-1]//2,bases[-2]//2]:
        actions[-3] = 1
    outputs = custom_seq_2_decimal(actions)
    return outputs
        


def action_2_token(inputs:tuple, tokenizer_type:str = 'llama-2'):
    '''
    Params: 
    inputs:tuple:两个十进制数字
    * output: str, 返回一个控制token
    Function: 生成一个控制token
    Examples:
    1. generate_control_token(15359) -> tuple(7,7,7,5,4) -> '단학중음백'
    2. generate_control_token(1) -> 'tuple(0,0,0,0,1) -> '현면만방재'
    '''
    # 生成控制token
    assert len(inputs)==2
    null_action = (0,60)
    custom_seq = decimal_2_custom_seq(inputs)
    zero_include_token_list = [map_control_token(num, i, tokenizer_type) for i, num in enumerate(custom_seq)]
    control_token = ''.join((s for x,s in zip(custom_seq[:-2],zero_include_token_list[:-2]) if x != 0 ))
    control_token = control_token + "".join((s for s in zero_include_token_list[-2:]))  #camera必须保存
    tag_control_token = tag_token(0,tokenizer_type) + control_token + tag_token(1,tokenizer_type)
    return tag_control_token,inputs==null_action

def decimal_2_custom_seq(inputs:tuple, bases:list = [10,3,3,3,2,2,2,2,11,11]) -> tuple:
    '''
    Params:
    * output: set, 返回一个元组, 元组中的每个元素表示一个位的值
    * inputs: tuple, 两个十进制整数
    * bases: list, 每位的基数     

    Function: 将一个十进制整数转换为具有不同基数的数字系统(每位的基数分别为 [8, 8, 8, 6, 5]), 需要编写一个Python函数来执行逆向计算。这个转换涉及将十进制数逐位除以对应的基数并取余数, 然后再继续处理商。
    Examples: 
    1. decimal_to_custom_base(1) -> (0, 0, 0, 0, 1)
    2. decimal_to_custom_base(15359) -> (7, 7, 7, 5, 4)
    '''
    decimals = list(inputs)
    decimals[0] = decimals[0]//2 #camera键在这里没有意义
    # 用于存储转换结果的列表
    result = [0] * len(bases)
    # 从最低位到最高位逐位计算
    for i in range(len(bases)-3, -1, -1):
        # 求当前位的值
        result[i] = decimals[0] % bases[i]
        # 更新十进制数为下一位的处理
        decimals[0] //= bases[i]
    # 确保转换过程中十进制数被完全转换
    result[-1] = decimals[1] % bases[-1]
    decimals[1] //= bases[-1]
    result[-2] = decimals[1] % bases[-2]
    decimals[1] //= bases[-2]

    if decimals != [0,0]:
        raise ValueError("The decimal number is too large for the custom base system.")
    return tuple(result)


def custom_seq_2_decimal(number_tuple:tuple, bases:list = [10,3,3,3,2,2,2,2,2,11,11]) -> tuple:
    '''
    假如bases为[10,3,3,3,2,2,2,2,2,11,11]
    Function: 将一个具有不同基数的数字系统(每位的基数分别为 [8, 8, 8, 6, 5])转换为十进制整数, 需要编写一个Python函数来执行逆向计算。这个转换涉及将每位的值乘以对应的基数的幂, 然后再求和。
    Examples:
    1. custom_base_to_decimal((0, 0, 0, 0, 1)) -> 1
    2. custom_base_to_decimal((7, 7, 7, 5, 4)) -> 15359
    :output: int, 十进制整数
    :number_tuple: tuple, 每位的值
    :bases: list, 每位的基数
    '''
    # 确保输入的长度与基数匹配
    if len(number_tuple) != len(bases):
        raise ValueError("The input number does not match the expected number of digits.")
    # 初始化十进制结果
    decimal_results = [0,0]
    # 计算十进制值
    mid = len(number_tuple)-2
    for i, digit in enumerate(number_tuple):
        if digit >= bases[i]:
            raise ValueError(f"Digit at position {i} exceeds the base limit of {bases[i]-1}.")
        if i < mid:
            decimal_results[0] = decimal_results[0] * bases[i] + digit
        else:
            decimal_results[1] = decimal_results[1] * bases[i] + digit
    return tuple(decimal_results)


class ActionMap:
    def __init__(self,tokenizer_type="llama-2",bases=[10,3,3,3,2,2,2,2,2,11,11]):
        self.tokenizer_type = tokenizer_type
        self.bases = bases
        self.action_transformer = ActionTransformer()
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
    
    def map(self,token):
        action = token_2_action(token,tokenizer_type=self.tokenizer_type,bases=self.bases)
        action_dict = {
            "buttons":np.array([action[0]]),
            "camera":np.array([action[1]]),
        }
        action_dict = OrderedDict({key: value[0] for key, value in action_dict.items()})
        #factored_action = self.action_mapper.to_factored(action_dict)
        #envir_action = self.action_transformer.policy2env(factored_action)
        return action_dict
    
if __name__ == "__main__":
    action_map = ActionMap("llama-3")
    print(action_map.map("<|reserved_special_token_199|><|reserved_special_token_234|><|reserved_special_token_245|><|reserved_special_token_198|>"))