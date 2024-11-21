import re
import numpy as np
from collections import OrderedDict
from typing import Union
import torch
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
    token_num = sum(bases)+30
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
mistral (llava-v1.6-mistral-7b-hf)
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
    ('摩', 31978),('袁', 31979),('ষ', 31980),('乎', 31981),('규', 31982),('岗', 31983),('糊', 31984),('క', 31985),('雲', 31986),('심', 31987),('ई', 31988),('庭', 31926), ('苗', 31927),('闲', 31929), ('독', 31930), ('ɹ', 31931), ('ҽ', 31932), ('ថ', 31933), ('宏', 31934), ('尊', 31935), ('總', 31936),
    ('འ', 31989),('ἡ', 31990),('丝', 31991),('Ħ', 31992),('ٍ', 31993),('ٓ', 31994),('အ', 31995),('執', 31996),('벨', 31997),('ゼ', 31998),('梦', 31999), ('裝', 31937), ('ම', 31938), ('▸', 31939), ('測', 31940), ('勇', 31920), ('徐', 31921), ('轩', 31943), ('兄', 31944), ('剑', 31945), ('ન', 31946)
]
llama-3,llava-next(llama3-llava-next-8b-hf)
[ 71 
    ('ĠìĦľìļ¸íĬ¹ë³Ħìĭľ', 127929), ('ÎķÎĻÎ£', 127930), ('à¸¸à¸¡à¸Ĭà¸Ļ', 127931), ('ĠÐ¼ÑĸÐ»ÑĮ', 127932), ('æħĮ', 127933), ('ÏĥÎºÎµÏĦÎ±Î¹', 127934), ('ĠãĢľ', 127935), ('Ġkaliteli', 127936), ('ĠÑģÐ¼ÐµÑĢÑĤÑĮ', 127937), ('è¼Ķ', 127938), 
    ('ĠÐ±Ð¸ÑĤ', 127939), ('ĠÎ£ÏĦÎ¿', 127940), ('à¸ĩà¹Ģà¸¨à¸ª', 127941), 
    ('åİŁæľ¬', 127942), ('ĠknÃŃ', 127943), ('äºĴèģĶç½ĳ', 127944), 
    ('ĠÑĩÐµÐ»Ð¾Ð²ÐµÑĩÐµÑģ', 127945), ('çŃĴ', 127946), ('à¸Īà¸³à¸«à¸Ļ', 127947), 
    ('åĩºåİ»', 127948), ('ãĤ¢ãĥĭãĥ¡', 127949), 
    ('å±ķç¤º', 127950), ('rych', 127951), 
    ('à¤ħà¤¬', 127952), ('oÅĪ', 127953), 
    ('jÃŃcÃŃm', 127954), ('Ø§ØŃØ«', 127955), 
    ('ĠÙĪØ§ÙĤØ¹ÛĮ', 127956), ('ĠÐ¤ÐµÐ´ÐµÑĢÐ°Ð»ÑĮ', 127957), 
    ('ÑģÐ°Ð¼', 127958), ('Ġìĺ¥', 127959), ('åľ°çĲĥ', 127960), ('Ġsuyu', 127961), ('seniz', 127962), ('à¥īà¤«', 127963), ('Ġê°Ļëĭ¤', 127964), ('ĠÐ¿ÑĢÐ¸Ð·Ð½Ð°ÑĩÐµÐ½Ð½Ñı', 127965), ('ĠSÄ±n', 127966), ('ĠØ§ÙħÙĨÛĮØª', 127967), ('ĠlÃ¡tky', 127968), ('ĠÐĳÐ¸', 127969), ('ĠsÃ¼reci', 127970), ('Â·Â·Â·Â·', 127971), ('Ġê²½ì°°', 127972), ('ĠÐºÐ°Ð»ÑĮ', 127973), ('ĠÐ½Ð¸ÐºÑĤÐ¾', 127974), ('ÙĳÙħ', 127975), ('ĠØ¯ÙĬÚ¯Ø±',127976), ('ĠalÄ±nmasÄ±', 127977), ('Ð»ÐµÐ½Ð½Ñĸ', 127978), 
    ('à¸´à¸§à¹Ģà¸ķà¸Ńà¸£', 127979), ('à¸Ľà¸ģà¸Ħà¸£à¸Ńà¸ĩ', 127980), ('ĠÐ·Ð°ÐºÐ¾Ð½Ð¾Ð´Ð°Ð²ÑģÑĤÐ²Ð°', 127981), ('ãĢĢãĤ¤', 127982), ('Ġëħ¸íķĺìļ°', 127983), ('ĠDÃ¼ÅŁ', 127984), ('ĠÐ³ÑĥÑģÑĤ', 127985), ('ĠÐĴÐ°ÑĪ', 127986), ('ĠØ§ÙħØªÛĮ', 127987), ('Ġparamet', 127988), ('ĠÎłÎ±Î½ÎµÏĢ', 127989), ('à¹Įà¸ģà¸£', 127990), ('Î¶Î±', 127991), ('ĠëįĶìļ±', 127992), ('ÙĪÙĦØ§Øª', 127993), ('Ð²Ð°ÑĤÐ¸ÑģÑı', 127994), ('ĠkÃ¶k', 127995), ('ÙĨØ¨', 127996), ('ĠÐ²ÑĭÑģÐ¾ÐºÐ¾Ð¹', 127997), ('ãĥ¼ãĥ¼', 127998), ('éĶ¦', 127999)
]
[ 128255-128187  68 52  61
     ('<|reserved_special_token_180|>', 128185), ('<|reserved_special_token_181|>', 128186), ('<|reserved_special_token_182|>', 128187), ('<|reserved_special_token_183|>', 128188), ('<|reserved_special_token_184|>', 128189), ('<|reserved_special_token_185|>', 128190), ('<|reserved_special_token_186|>', 128191), ('<|reserved_special_token_187|>', 128192), ('<|reserved_special_token_188|>', 128193), ('<|reserved_special_token_189|>', 128194), 
     ('<|reserved_special_token_190|>', 128195), ('<|reserved_special_token_191|>', 128196), ('<|reserved_special_token_192|>', 128197), 
     ('<|reserved_special_token_193|>', 128198), ('<|reserved_special_token_194|>', 128199), ('<|reserved_special_token_195|>', 128200), 
     ('<|reserved_special_token_196|>', 128201), ('<|reserved_special_token_197|>', 128202), ('<|reserved_special_token_198|>', 128203), 
     ('<|reserved_special_token_199|>', 128204), ('<|reserved_special_token_200|>', 128205),
     ('<|reserved_special_token_201|>', 128206), ('<|reserved_special_token_202|>', 128207), 
     ('<|reserved_special_token_203|>', 128208), ('<|reserved_special_token_204|>', 128209), 
     ('<|reserved_special_token_205|>', 128210), ('<|reserved_special_token_206|>', 128211),
     ('<|reserved_special_token_207|>', 128212), ('<|reserved_special_token_208|>', 128213), 
     ('<|reserved_special_token_209|>', 128214), ('<|reserved_special_token_210|>', 128215), ('<|reserved_special_token_211|>', 128216), ('<|reserved_special_token_212|>', 128217), ('<|reserved_special_token_213|>', 128218), ('<|reserved_special_token_214|>', 128219), ('<|reserved_special_token_215|>', 128220), ('<|reserved_special_token_216|>', 128221), ('<|reserved_special_token_217|>', 128222), ('<|reserved_special_token_218|>', 128223), ('<|reserved_special_token_219|>', 128224), ('<|reserved_special_token_220|>', 128225), ('<|reserved_special_token_221|>', 128226), ('<|reserved_special_token_222|>', 128227), ('<|reserved_special_token_223|>', 128228), ('<|reserved_special_token_224|>', 128229), ('<|reserved_special_token_225|>', 128230), ('<|reserved_special_token_226|>', 128231), ('<|reserved_special_token_227|>', 128232), ('<|reserved_special_token_228|>', 128233), ('<|reserved_special_token_229|>', 128234), 
     ('<|reserved_special_token_230|>', 128235), ('<|reserved_special_token_231|>', 128236), ('<|reserved_special_token_232|>', 128237), ('<|reserved_special_token_233|>', 128238), ('<|reserved_special_token_234|>', 128239), ('<|reserved_special_token_235|>', 128240), ('<|reserved_special_token_236|>', 128241), ('<|reserved_special_token_237|>', 128242), ('<|reserved_special_token_238|>', 128243), ('<|reserved_special_token_239|>', 128244), ('<|reserved_special_token_240|>', 128245), ('<|reserved_special_token_241|>', 128246), ('<|reserved_special_token_242|>', 128247), ('<|reserved_special_token_243|>', 128248), ('<|reserved_special_token_244|>', 128249), ('<|reserved_special_token_245|>', 128250), ('<|reserved_special_token_246|>', 128251), ('<|reserved_special_token_247|>', 128252), ('<|reserved_special_token_248|>', 128253), ('<|reserved_special_token_249|>', 128254), ('<|reserved_special_token_250|>', 128255)
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
            (('개', 31789), ('들', 31804),),
            (('차', 31817), ('학', 31822), ('만', 31826), ('터', 31856), ('식', 31895), ('과', 31906), ('타', 31925), ('종', 31930), ('내', 31940), ('중', 31941), ('방', 31945)),
            (('월', 31950), ('회', 31953), ('모', 31962), ('바', 31963), ('음', 31966), ('재', 31973), ('명', 31976), ('합', 31980), ('역', 31987), ('백', 31989), ('왕', 31996)),
        ]
    elif tokenizer_type == "mistral":
        special_tokens = [
            (('朱', 31947),('ǝ', 31948),('Ḩ', 31949),('担', 31950),('灰', 31951), ('讲', 31952), ('롤', 31953),('😤', 31955),('ោ', 31956),('애', 31957),),
            (('였', 31958),('질', 31959),('振', 31960),),
            (('灯', 31961),('ĉ', 31962),('ස', 31963),),
            (('閉', 31964),('램', 31965),('ಂ', 31966),),
            (('げ', 31967),('ふ', 31896),),
            (('狂', 31969),('融', 31970),),
            (('仍', 31971),('實', 31972),),
            (('楽', 31973),('範', 31974),),
            (('వ', 31976),('嵌', 31977),),
            (('摩', 31978),('袁', 31979),('ষ', 31980),('乎', 31981),('규', 31982),('岗', 31983),('糊', 31984),('క', 31985),('雲', 31986),('심', 31987),('ई', 31988),('庭', 31926), ('苗', 31927),('闲', 31929), ('독', 31930), ('ɹ', 31931), ('ҽ', 31932), ('ថ', 31933), ('宏', 31934), ('尊', 31935), ('總', 31936),),
            (('འ', 31989),('ἡ', 31990),('丝', 31991),('Ħ', 31992),('ٍ', 31993),('ٓ', 31994),('အ', 31995),('執', 31996),('벨', 31997),('ゼ', 31998),('梦', 31999), ('裝', 31937), ('ම', 31938), ('▸', 31939), ('測', 31940), ('勇', 31920), ('徐', 31921), ('轩', 31943), ('兄', 31944), ('剑', 31945), ('ન', 31946),),
        ]
    elif tokenizer_type == "llama-3":
        special_tokens = [
            (('<|reserved_special_token_180|>', 128185), ('<|reserved_special_token_181|>', 128186), ('<|reserved_special_token_182|>', 128187), ('<|reserved_special_token_183|>', 128188), ('<|reserved_special_token_184|>', 128189), ('<|reserved_special_token_185|>', 128190), ('<|reserved_special_token_186|>', 128191), ('<|reserved_special_token_187|>', 128192), ('<|reserved_special_token_188|>', 128193), ('<|reserved_special_token_189|>', 128194), ),
            (('<|reserved_special_token_190|>', 128195), ('<|reserved_special_token_191|>', 128196), ('<|reserved_special_token_192|>', 128197), ),
            (('<|reserved_special_token_193|>', 128198), ('<|reserved_special_token_194|>', 128199), ('<|reserved_special_token_195|>', 128200), ),
            (('<|reserved_special_token_196|>', 128201), ('<|reserved_special_token_197|>', 128202), ('<|reserved_special_token_198|>', 128203), ),
            (('<|reserved_special_token_199|>', 128204), ('<|reserved_special_token_200|>', 128205),),
            (('<|reserved_special_token_201|>', 128206), ('<|reserved_special_token_202|>', 128207), ),
            (('<|reserved_special_token_203|>', 128208), ('<|reserved_special_token_204|>', 128209), ),
            (('<|reserved_special_token_205|>', 128210), ('<|reserved_special_token_206|>', 128211),),
            (('<|reserved_special_token_207|>', 128212), ('<|reserved_special_token_208|>', 128213), ),
            (('<|reserved_special_token_209|>', 128214), ('<|reserved_special_token_210|>', 128215), ('<|reserved_special_token_211|>', 128216), ('<|reserved_special_token_212|>', 128217), ('<|reserved_special_token_213|>', 128218), ('<|reserved_special_token_214|>', 128219), ('<|reserved_special_token_215|>', 128220), ('<|reserved_special_token_216|>', 128221), ('<|reserved_special_token_217|>', 128222), ('<|reserved_special_token_218|>', 128223), ('<|reserved_special_token_219|>', 128224), ('<|reserved_special_token_220|>', 128225), ('<|reserved_special_token_221|>', 128226), ('<|reserved_special_token_222|>', 128227), ('<|reserved_special_token_223|>', 128228), ('<|reserved_special_token_224|>', 128229), ('<|reserved_special_token_225|>', 128230), ('<|reserved_special_token_226|>', 128231), ('<|reserved_special_token_227|>', 128232), ('<|reserved_special_token_228|>', 128233), ('<|reserved_special_token_229|>', 128234), ),
            (('<|reserved_special_token_230|>', 128235), ('<|reserved_special_token_231|>', 128236), ('<|reserved_special_token_232|>', 128237), ('<|reserved_special_token_233|>', 128238), ('<|reserved_special_token_234|>', 128239), ('<|reserved_special_token_235|>', 128240), ('<|reserved_special_token_236|>', 128241), ('<|reserved_special_token_237|>', 128242), ('<|reserved_special_token_238|>', 128243), ('<|reserved_special_token_239|>', 128244), ('<|reserved_special_token_240|>', 128245), ('<|reserved_special_token_241|>', 128246), ('<|reserved_special_token_242|>', 128247), ('<|reserved_special_token_243|>', 128248), ('<|reserved_special_token_244|>', 128249), ('<|reserved_special_token_245|>', 128250), ('<|reserved_special_token_246|>', 128251), ('<|reserved_special_token_247|>', 128252), ('<|reserved_special_token_248|>', 128253), ('<|reserved_special_token_249|>', 128254), ('<|reserved_special_token_250|>', 128255)),
        ]
        special_tokens1 = [
            (('ĠìĦľìļ¸íĬ¹ë³Ħìĭľ', 127929), ('ÎķÎĻÎ£', 127930), ('à¸¸à¸¡à¸Ĭà¸Ļ', 127931), ('ĠÐ¼ÑĸÐ»ÑĮ', 127932), ('æħĮ', 127933), ('ÏĥÎºÎµÏĦÎ±Î¹', 127934), ('ĠãĢľ', 127935), ('Ġkaliteli', 127936), ('ĠÑģÐ¼ÐµÑĢÑĤÑĮ', 127937), ('è¼Ķ', 127938),),
            (('ĠÐ±Ð¸ÑĤ', 127939), ('ĠÎ£ÏĦÎ¿', 127940), ('à¸ĩà¹Ģà¸¨à¸ª', 127941), ),
            (('åİŁæľ¬', 127942), ('ĠknÃŃ', 127943), ('äºĴèģĶç½ĳ', 127944), ),
            (('ĠÑĩÐµÐ»Ð¾Ð²ÐµÑĩÐµÑģ', 127945), ('çŃĴ', 127946), ('à¸Īà¸³à¸«à¸Ļ', 127947), ),
            (('åĩºåİ»', 127948), ('ãĤ¢ãĥĭãĥ¡', 127949), ),
            (('å±ķç¤º', 127950), ('rych', 127951), ),
            (('à¤ħà¤¬', 127952), ('oÅĪ', 127953), ),
            (('jÃŃcÃŃm', 127954), ('Ø§ØŃØ«', 127955), ),
            (('ĠÙĪØ§ÙĤØ¹ÛĮ', 127956), ('ĠÐ¤ÐµÐ´ÐµÑĢÐ°Ð»ÑĮ', 127957), ),
            (('ÑģÐ°Ð¼', 127958), ('Ġìĺ¥', 127959), ('åľ°çĲĥ', 127960), ('Ġsuyu', 127961), ('seniz', 127962), ('à¥īà¤«', 127963), ('Ġê°Ļëĭ¤', 127964), ('ĠÐ¿ÑĢÐ¸Ð·Ð½Ð°ÑĩÐµÐ½Ð½Ñı', 127965), ('ĠSÄ±n', 127966), ('ĠØ§ÙħÙĨÛĮØª', 127967), ('ĠlÃ¡tky', 127968), ('ĠÐĳÐ¸', 127969), ('ĠsÃ¼reci', 127970), ('Â·Â·Â·Â·', 127971), ('Ġê²½ì°°', 127972), ('ĠÐºÐ°Ð»ÑĮ', 127973), ('ĠÐ½Ð¸ÐºÑĤÐ¾', 127974), ('ÙĳÙħ', 127975), ('ĠØ¯ÙĬÚ¯Ø±',127976), ('ĠalÄ±nmasÄ±', 127977), ('Ð»ÐµÐ½Ð½Ñĸ', 127978),),
            (('à¸´à¸§à¹Ģà¸ķà¸Ńà¸£', 127979), ('à¸Ľà¸ģà¸Ħà¸£à¸Ńà¸ĩ', 127980), ('ĠÐ·Ð°ÐºÐ¾Ð½Ð¾Ð´Ð°Ð²ÑģÑĤÐ²Ð°', 127981), ('ãĢĢãĤ¤', 127982), ('Ġëħ¸íķĺìļ°', 127983), ('ĠDÃ¼ÅŁ', 127984), ('ĠÐ³ÑĥÑģÑĤ', 127985), ('ĠÐĴÐ°ÑĪ', 127986), ('ĠØ§ÙħØªÛĮ', 127987), ('Ġparamet', 127988), ('ĠÎłÎ±Î½ÎµÏĢ', 127989), ('à¹Įà¸ģà¸£', 127990), ('Î¶Î±', 127991), ('ĠëįĶìļ±', 127992), ('ÙĪÙĦØ§Øª', 127993), ('Ð²Ð°ÑĤÐ¸ÑģÑı', 127994), ('ĠkÃ¶k', 127995), ('ÙĨØ¨', 127996), ('ĠÐ²ÑĭÑģÐ¾ÐºÐ¾Ð¹', 127997), ('ãĥ¼ãĥ¼', 127998), ('éĶ¦', 127999))
        ]
    else:
        raise ValueError(f"The tokenizer type {tokenizer_type} is not supported in control tokens.")
    #print(place,num,not_text)
    return special_tokens[place][num][not_text]

def prepare_for_remap_control_token(tokenizer_type:str = "llama-2",bases:list = [10,3,3,3,2,2,2,2,2,21,21],not_text=True):
    
    tokens = {}
    for i,base in enumerate(bases):
        for j in range(base):
            print(j,i,tokenizer_type)
            token = map_control_token(j,i,tokenizer_type,not_text=not_text)
            tokens[token]=(i,j)
    return tokens

def remap_control_token(token:str,use_num=True, tokenizer_type:str = "llama-2")->tuple:
    """由token映射到action，注意，虽然把camera从token中去掉，但是还需要它 """
    re_tokens = {}
    if tokenizer_type == "llama-2":
        if use_num:
            re_tokens = {31536:(0, 0), 31537:(0, 1), 31563:(0, 2), 31571:(0, 3), 31578:(0, 4), 31582:(0, 5), 31585:(0, 6), 31598:(0, 7), 31603:(0, 8), 31604:(0, 9), 31607:(1, 0), 31609:(1, 1), 31614:(1, 2), 31621:(2, 0), 31622:(2, 1), 31680:(2, 2), 31699:(3, 0), 31716:(3, 1), 31724:(3, 2), 31734:(4, 0), 31736:(4, 1), 31737:(5, 0), 31746:(5, 1), 31747:(6, 0), 31754:(6, 1), 31774:(7, 0), 31781:(7, 1), 31789:(8, 0), 31804:(8, 1), 31817:(9, 0), 31822:(9, 1), 31826:(9, 2), 31856:(9, 3), 31895:(9, 4), 31906:(9, 5), 31925:(9, 6), 31930:(9, 7), 31940:(9, 8), 31941:(9, 9), 31945:(9, 10), 31950:(10, 0), 31953:(10, 1), 31962:(10, 2), 31963:(10, 3), 31966:(10, 4), 31973:(10, 5), 31976:(10, 6), 31980:(10, 7), 31987:(10, 8), 31989:(10, 9), 31996:(10, 10)}
        else:
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
    elif tokenizer_type=="mistral":
        if use_num:
            re_tokens = {31947: (0, 0), 31948: (0, 1), 31949: (0, 2), 31950: (0, 3), 31951: (0, 4), 31952: (0, 5), 31953: (0, 6), 31955: (0, 7), 31956: (0, 8), 31957: (0, 9), 31958: (1, 0), 31959: (1, 1), 31960: (1, 2), 31961: (2, 0), 31962: (2, 1), 31963: (2, 2), 31964: (3, 0), 31965: (3, 1), 31966: (3, 2), 31967: (4, 0), 31896: (4, 1), 31969: (5, 0), 31970: (5, 1), 31971: (6, 0), 31972: (6, 1), 31973: (7, 0), 31974: (7, 1), 31976: (8, 0), 31977: (8, 1), 31978: (9, 0), 31979: (9, 1), 31980: (9, 2), 31981: (9, 3), 31982: (9, 4), 31983: (9, 5), 31984: (9, 6), 31985: (9, 7), 31986: (9, 8), 31987: (9, 9), 31988: (9, 10), 31926: (9, 11), 31927: (9, 12), 31929: (9, 13), 31930: (9, 14), 31931: (9, 15), 31932: (9, 16), 31933: (9, 17), 31934: (9, 18), 31935: (9, 19), 31936: (9, 20), 31989: (10, 0), 31990: (10, 1), 31991: (10, 2), 31992: (10, 3), 31993: (10, 4), 31994: (10, 5), 31995: (10, 6), 31996: (10, 7), 31997: (10, 8), 31998: (10, 9), 31999: (10, 10), 31937: (10, 11), 31938: (10, 12), 31939: (10, 13), 31940: (10, 14), 31920: (10, 15), 31921: (10, 16), 31943: (10, 17), 31944: (10, 18), 31945: (10, 19), 31946: (10, 20)}
        else:
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
                '摩': (9, 0),'袁': (9, 1),'ষ': (9, 2),'乎': (9, 3),'규': (9, 4),'岗': (9, 5),'糊': (9, 6),'క': (9, 7),'雲': (9, 8),'심': (9, 9),'ई': (9, 10),'庭': (9, 11), '苗': (9, 12), '闲': (9, 13), '독': (9, 14),'ɹ': (9, 15), 'ҽ': (9, 16), 'ថ': (9, 17), '宏': (9, 18), '尊': (9, 19),'總': (9, 20),
                'འ': (10, 0),'ἡ': (10, 1),'丝': (10, 2),'Ħ': (10, 3),'伝': (10, 4),'컨': (10, 5),'အ': (10, 6),'執': (10, 7),'벨': (10, 8),'ゼ': (10, 9),'梦': (10, 10),'裝': (10, 11), 'ම': (10, 12), '▸': (10, 13), '測': (10, 14),'勇': (10, 15), '徐': (10, 16), '轩': (10, 17), '兄': (10, 18), '剑': (10, 19),'ન': (10, 20)
            }
    elif tokenizer_type=="llama-3":
        if use_num:
            re_tokens={128185: (0, 0), 128186: (0, 1), 128187: (0, 2), 128188: (0, 3), 128189: (0, 4), 128190: (0, 5), 128191: (0, 6), 128192: (0, 7), 128193: (0, 8), 128194: (0, 9), 128195: (1, 0), 128196: (1, 1), 128197: (1, 2), 128198: (2, 0), 128199: (2, 1), 128200: (2, 2), 128201: (3, 0), 128202: (3, 1), 128203: (3, 2), 128204: (4, 0), 128205: (4, 1), 128206: (5, 0), 128207: (5, 1), 128208: (6, 0), 128209: (6, 1), 128210: (7, 0), 128211: (7, 1), 128212: (8, 0), 128213: (8, 1), 128214: (9, 0), 128215: (9, 1), 128216: (9, 2), 128217: (9, 3), 128218: (9, 4), 128219: (9, 5), 128220: (9, 6), 128221: (9, 7), 128222: (9, 8), 128223: (9, 9), 128224: (9, 10), 128225: (9, 11), 128226: (9, 12), 128227: (9, 13), 128228: (9, 14), 128229: (9, 15), 128230: (9, 16), 128231: (9, 17), 128232: (9, 18), 128233: (9, 19), 128234: (9, 20), 128235: (10, 0), 128236: (10, 1), 128237: (10, 2), 128238: (10, 3), 128239: (10, 4), 128240: (10, 5), 128241: (10, 6), 128242: (10, 7), 128243: (10, 8), 128244: (10, 9), 128245: (10, 10), 128246: (10, 11), 128247: (10, 12), 128248: (10, 13), 128249: (10, 14), 128250: (10, 15), 128251: (10, 16), 128252: (10, 17), 128253: (10, 18), 128254: (10, 19), 128255: (10, 20)}
            re_tokens1={127929: (0, 0), 127930: (0, 1), 127931: (0, 2), 127932: (0, 3), 127933: (0, 4), 127934: (0, 5), 127935: (0, 6), 127936: (0, 7), 127937: (0, 8), 127938: (0, 9), 
                       127939: (1, 0), 127940: (1, 1), 127941: (1, 2),
                       127942: (2, 0), 127943: (2, 1), 127944: (2, 2),
                       127945: (3, 0), 127946: (3, 1), 127947: (3, 2), 
                       127948: (4, 0), 127949: (4, 1), 
                       127950: (5, 0), 127951: (5, 1), 
                       127952: (6, 0), 127953: (6, 1), 
                       127954: (7, 0), 127955: (7, 1), 
                       127956: (8, 0), 127957: (8, 1), 
                       127958: (9, 0), 127959: (9, 1), 127960: (9, 2), 127961: (9, 3), 127962: (9, 4), 127963: (9, 5), 127964: (9, 6), 127965: (9, 7), 127966: (9, 8), 127967: (9, 9), 127968: (9, 10), 127969: (9, 11), 127970: (9, 12), 127971: (9, 13), 127972: (9, 14), 127973: (9, 15), 127974: (9, 16), 127975: (9, 17), 127976: (9, 18), 127977: (9, 19), 127978: (9, 20), 
                       127979: (10, 0), 127980: (10, 1), 127981: (10, 2), 127982: (10, 3), 127983: (10, 4), 127984: (10, 5), 127985: (10, 6), 127986: (10, 7), 127987: (10, 8), 127988: (10, 9), 127989: (10, 10), 127990: (10, 11), 127991: (10, 12), 127992: (10, 13), 127993: (10, 14), 127994: (10, 15), 127995: (10, 16), 127996: (10, 17), 127997: (10, 18), 127998: (10, 19), 127999: (10, 20)
                    }
        else:
            raise ValueError("can't use text as tokens")
    else:
        raise ValueError(f"The tokenizer type {tokenizer_type} is not supported in control tokens.")
    return re_tokens.get(token,(-1,-1))

def tag_token(place, tokenizer_type:str = "llama-2",return_type:int=0):
    """引入头标记和尾标记 """
    assert place in {0,1}
    if tokenizer_type == "llama-2":
        special_tokens = [('유', 31533),('요', 31527)]
    elif tokenizer_type == "mistral":
        special_tokens = [('ಮ', 31941),('አ', 31942)]
    elif tokenizer_type=="llama-3":
        special_tokens = [('<|reserved_special_token_178|>', 128183), ('<|reserved_special_token_179|>', 128184),]
        special_tokens1 = [('poÄįet', 127927), ('ë§ĮìĽĲìŀħëĭĪëĭ¤', 127928)]
    else:
        raise ValueError(f"The tokenizer type {tokenizer_type} is not supported in control tokens.")
    return special_tokens[place][return_type]

def token_2_action(tokens:Union[str,torch.Tensor],tag_token_list, tokenizer_type:str = 'llama-2',bases:list = [10,3,3,3,2,2,2,2,2,21,21]) -> tuple:
    """将一个输入序列转换回 """
    actions = [0]*len(bases) #初始化
    camera_null = [bases[-1]//2,bases[-2]//2]
    actions[-2:] = camera_null
    if isinstance(tokens,str):
        #输入文字
        pattern = f'{tag_token(0,tokenizer_type)}.*{tag_token(1,tokenizer_type)}'
        match = re.search(pattern, tokens)
        
        if not match:
            return custom_seq_2_decimal(actions,bases=bases)
        control_tokens = match.group()[1:-1]
        for token in control_tokens:
            place,num = remap_control_token(token,use_num=False,tokenizer_type=tokenizer_type)
            if place!=-1:
                actions[place]=num
    elif isinstance(tokens,torch.Tensor):

        if tokens.shape == 2:
            tokens = tokens.squeeze()
        indices_n1 = torch.where(tokens == tag_token_list[0])
        first_index_n1 = indices_n1[0][0].item() if indices_n1[0].numel() > 0 else None

        indices_n2 = torch.where(tokens == tag_token_list[1])
        first_index_n2 = indices_n2[0][0].item() if indices_n2[0].numel() > 0 else None

        if first_index_n1 is not None and first_index_n2 is not None and first_index_n1 < first_index_n2:
            control_tokens = tokens[first_index_n1 + 1:first_index_n2]
        else:
            return custom_seq_2_decimal(actions,bases=bases)
        for token in control_tokens:
            place,num = remap_control_token(token.item(),use_num=True,tokenizer_type=tokenizer_type)
            if place!=-1:
                actions[place]=num
    elif isinstance(tokens, list):
    # 查找匹配 tag_token_list[0] 的所有索引
        indices_n1 = [i for i, token in enumerate(tokens) if token == tag_token_list[0]]
        first_index_n1 = indices_n1[0] if indices_n1 else None
        indices_n2 = [i for i, token in enumerate(tokens) if token == tag_token_list[1]]
        first_index_n2 = indices_n2[0] if indices_n2 else None
        if first_index_n1 is not None and first_index_n2 is not None and first_index_n1 < first_index_n2:
            control_tokens = tokens[first_index_n1 + 1:first_index_n2]
        else:
            return custom_seq_2_decimal(actions, bases=bases)
        for token in control_tokens:
            place, num = remap_control_token(token, use_num=True, tokenizer_type=tokenizer_type)
            if place != -1:
                actions[place] = num
    else:
        raise ValueError("wrong type!")
    # 如果移动了视野，camera button变为1
    if actions[-2:] != camera_null:
        actions[-3] = 1
    outputs = custom_seq_2_decimal(actions,bases=bases)
    return outputs
        
def action_2_token(inputs:tuple, tokenizer_type:str = 'llama-2', bases:list = [10,3,3,3,2,2,2,2,2,21,21]):
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
    null_action = (0,bases[-1]*(bases[-2]//2)+bases[-1]//2)
    custom_seq = decimal_2_custom_seq(inputs,bases=bases)
    zero_include_token_list = [map_control_token(num, i, tokenizer_type) for i, num in enumerate(custom_seq)]
    control_token = ''.join((s for x,s in zip(custom_seq[:-3],zero_include_token_list[:-3]) if x != 0 )) #camera键在这里没有意义
    control_token = control_token + "".join((s for s in zero_include_token_list[-2:]))  #camera必须保存
    tag_control_token = tag_token(0,tokenizer_type) + control_token + tag_token(1,tokenizer_type)
    return tag_control_token,inputs==null_action

def decimal_2_custom_seq(inputs:tuple, bases:list = [10,3,3,3,2,2,2,2,2,21,21]) -> tuple:
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
    #decimals[0] = decimals[0]//2 #camera键在这里没有意义
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


def custom_seq_2_decimal(number_tuple:tuple, bases:list = [10,3,3,3,2,2,2,2,2,21,21]) -> tuple:
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


class ActionTokenizer:
    def __init__(self,tokenizer_type="llama-2",bases=[10,3,3,3,2,2,2,2,2,21,21]):
        from rich import console
        self.tokenizer_type = tokenizer_type
        self.bases = bases
        self.NULL_ACTION = [0,0]
        self.NULL_ACTION[-1] = (bases[-2]//2)*bases[-2]+(bases[-1]//2)
        console.Console().log(f"warning, 修改了bases:{bases}")
        self.action_transformer = ActionTransformer(camera_quantization_scheme="mu_law",camera_mu=20,camera_binsize=1)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=21)
        self.basic_tag_token = [tag_token(0,self.tokenizer_type,return_type=1),tag_token(1,self.tokenizer_type,return_type=1)]
    
    def map(self,tokens):
        action = token_2_action(tokens,tag_token_list=self.basic_tag_token,tokenizer_type=self.tokenizer_type,bases=self.bases)
        action_dict = {
            "buttons":np.array([action[0]]),
            "camera":np.array([action[1]]),  #返回一个工作
        }
        action_dict = OrderedDict({key: value[0] for key, value in action_dict.items()})
        #factored_action = self.action_mapper.to_factored(action_dict)
        #envir_action = self.action_transformer.policy2env(factored_action)
        return action_dict
    
    def token2action(self,tokens):
        return self.map(tokens)
    
    def action2token(self,action:list):
        return action_2_token(action,tokenizer_type=self.tokenizer_type,bases=self.bases)
    
    def null_token(self):
        return self.action2token(self.NULL_ACTION)
    
if __name__ == "__main__":
    #print(get_special_token("/nfs-shared/models/llama3-llava-next-8b-hf",bases = [10, 3, 3, 3, 2, 2, 2, 2, 2, 21, 500]))
    #print(prepare_for_remap_control_token("llama-3",not_text=False))
    #exit()
    #print(action_2_token(inputs=(1,1),tokenizer_type="mistral"))
    outp = token_2_action(tokens=torch.tensor([127927,127983,127999,127928]),tag_token_list=[127927,127928],tokenizer_type="llama-3")
    print(outp)
    #exit()
    bases=[11,11]
    #print(bases[-1]*(bases[-2]//2)+bases[-1]//2)
    #print(prepare_for_remap_control_token(tokenizer_type="mistral"))
    #print(get_special_token("/nfs-shared/models/llava-v1.6-mistral-7b-hf",bases = [10, 3, 3, 3, 2, 2, 2, 2, 2, 21, 21]))
    action_map = ActionTokenizer("llama-3")
    print(action_map.map(tokens=torch.tensor([127927,127930,127983,127999,127928])))