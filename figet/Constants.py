
PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = "<blank>"
UNK_WORD = "unk"
BOS_WORD = "<s>"
EOS_WORD = "</s>"

BUFFER_SIZE = 64 * (1024 ** 2)

TOKEN_VOCAB = "token"
TYPE_VOCAB = "type"
CHAR_VOCAB = "char"


# dataset file fields
MENTION = "mention_span"
RIGHT_CTX = "right_context_token"
LEFT_CTX = "left_context_token"
TYPE = "y_str"

EPS = 1e-5

COARSE_FLAG = 0
FINE_FLAG = 1
UF_FLAG = 2

COARSE = {'person', 'group', 'organization', 'location', 'entity', 'time', 'object', 'event', 'place'}
FINE = {'accident', 'actor', 'agency', 'airline', 'airplane', 'airport', 'animal', 'architect', 'army', 'art',
        'artist', 'athlete', 'attack', 'author', 'award', 'biology', 'body_part', 'bridge', 'broadcast',
        'broadcast_station', 'building', 'car', 'cemetery', 'chemistry', 'city', 'coach', 'company', 'computer',
        'conflict', 'country', 'county', 'currency', 'degree', 'department', 'director', 'disease', 'doctor', 'drug',
        'education', 'election', 'engineer', 'ethnic_group', 'facility', 'film', 'finance', 'food', 'game', 'geography',
        'god', 'government', 'health', 'heritage', 'holiday', 'hospital', 'hotel', 'institution', 'instrument',
        'internet', 'island', 'language', 'law', 'lawyer', 'league', 'leisure', 'library', 'living_thing',
        'mass_transit', 'medicine', 'military', 'mobile_phone', 'monarch', 'mountain', 'music', 'musician',
        'music_school', 'natural_disaster', 'news', 'news_agency', 'park', 'planet', 'play', 'political_party',
        'politician', 'product', 'programming_language', 'protest', 'province', 'rail', 'railway', 'religion',
        'religious_leader', 'restaurant', 'road', 'scientific_method', 'ship', 'sign', 'society', 'software', 'soldier',
        'spacecraft', 'sport', 'stage', 'stock_exchange', 'structure', 'subway', 'team', 'television_channel',
        'television_network', 'television_program', 'theater', 'title', 'train', 'transit', 'transportation',
        'treatment', 'water', 'weapon', 'website', 'writing'}


CHARS = ['!', '"', '#', '$', '%', '&', "'", '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
         '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
         'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd',
         'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
         '{', '}', '~', '·', 'Ì', 'Û', 'à', 'ò', 'ö', '˙', 'ِ', '’', '→', '■', '□', '●', '【', '】', 'の', '・', '一', '（',
         '）', '＊', '：', '￥']
