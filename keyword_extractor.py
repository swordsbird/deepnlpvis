from utils import word_ordering
from config import max_keyword_num


def get_sentences_keywords(loader, idxs, layer):
    if layer == -1:
        layer = loader.n_layer - 1
    represents = {}
    norm_contri = {}
    contri = {}
    tf = {}
    for i in idxs:
        sent = loader.word_info_by_layer[layer][i]
        for p in sent:
            stem_word, word, contri0, norm_contri0 = p
            if stem_word not in represents:
                represents[stem_word] = {}
            represents[stem_word][word] = represents[stem_word].get(
                word, 0) + 1
            tf[stem_word] = tf.get(stem_word, 0) + 1
            norm_contri[stem_word] = norm_contri.get(
                stem_word, 0) + norm_contri0
            contri[stem_word] = contri.get(stem_word, 0) + contri0
    for stem_word in norm_contri:
        norm_contri[stem_word] = norm_contri[stem_word] / tf[stem_word]
        contri[stem_word] = contri[stem_word] / tf[stem_word]

    tmp = []
    for stem_word in represents:
        max_count = 0
        represent_word = ''
        for word in represents[stem_word]:
            if represents[stem_word][word] > max_count:
                max_count = represents[stem_word][word]
                represent_word = word
        tmp.append((stem_word, represent_word))
    represents = dict(tmp)

    words = [(represents[key], norm_contri[key], contri[key], tf[key])
             for key in norm_contri if represents[key] not in loader.stop_words]
    words = word_ordering(words)

    ret = []
    for x in words[:max_keyword_num]:
        word, importance, tf, contri, _ = x
        ret.append({
            'word': word,
            'value': round(importance, 4),
            'frequency': round(tf, 4),
            'entropy': round(contri, 4),
            'contri': loader.layer_contri[-1][word],
            'score': loader.word_prediction_score[word],
            'embedding': loader.word_embedding[word],
        })
    return ret
