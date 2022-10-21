from model.Text_Classification import TS4CS
from model.Name_Entity_Recognition import NER4CS

# TS4CS('data/input/1625740876000_posts_post_cls.json')

output = NER4CS(TS4CS('data/input/1625740876000_posts_post_cls.json'))
for i in output:
    print(i)