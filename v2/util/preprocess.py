import emoji  # type: ignore

import re
RE_EMOJI = re.compile(r'[\u263a-\U0001f973]')
RE_MENTION = re.compile(r'<a[^>]*>(.*?)</a>')
RE_HTML_TAG = re.compile(r'<.*?>')
RE_HASHTAG = re.compile(r'#([^\W_]{1,50})', re.U)
RE_URL_1 = re.compile(r'http\S+')
RE_URL_2 = re.compile(r'www.+')
RE_REPETITION_CHAR = re.compile(r'([a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ])\1*', re.U)
RE_REPETITION_CHAR_OLD = re.compile(r'([\w])\1*')
RE_REPETITION_DASH = re.compile(r'([-.])\1*', re.U)


def mask_emoji(text,
               convert=False,
               delimiters=(" ", " "),
               language='vi'):
    """Mask emoji from text.

    Args:
        text (str): text contain emojies
        convert (bool, optional): Convert emojies to text instead \
            of removing. Defaults to False.
        language (str): Choose language of emoji name. Default to 'vi'.
        delimiters (tuple, optional): Delimiter for emojies. \
            Used when convert=True. Default to ("", "").

    Returns:
        str: text without graphical emojies
    """
    if convert:
        text = emoji.demojize(text, delimiters=delimiters, language=language)
    else:
        text = emoji.get_emoji_regexp().sub(r'', text)

    return text.strip()


def mask_mention(text, repl_string="@USER"):
    text = re.sub(RE_MENTION, repl_string, text)
    return text.strip()


def mask_url(text, repl_string="HTTPURL"):
    text = re.sub(RE_URL_1, repl_string, text)
    text = re.sub(RE_URL_2, repl_string, text)
    return text.strip()


def replace_br(text):
    text = text.replace("<br>", ".")
    return text


def remove_html(text):
    text = re.sub(RE_HTML_TAG, r'', text)
    return text.strip()


def remove_hashtag(text):
    text = re.sub(RE_HASHTAG, r'', text)
    return text.strip()


def remove_repetition(text, target="dash"):
    if target == "dash":
        text = re.sub(RE_REPETITION_DASH, r'\1', text)    
    else:
        text = re.sub(RE_REPETITION_CHAR, r'\1', text)
    return text


# if __name__ == "__main__":
#     print(
#         mask_emoji(r'🎁TẶNG QUÀ & SĂN KHUYẾN MÃI TỚI 50% KHI XEM LIVESTREAM \
#                     CÙNG 7-ELEVEN🔥', True))
#     print(
#         mask_emoji(r'🎁TẶNG QUÀ & SĂN KHUYẾN MÃI TỚI 50% KHI XEM LIVESTREAM \
#                     CÙNG 7-ELEVEN🔥'))

#     print(
#         mask_mention(r'<a href="/tran.pham.334839?refid=52&amp;__tn__=R">\
#             Trân Phạm</a> mai t giả bộ lên m làm bài r mua mấy này ăn'))

#     print(
#         mask_mention(r'<a href="/tran.pham.334839?refid=52&amp;__tn__=R">\
#             Trân Phạm</a> mai t giả bộ lên m làm bài r mua mấy này ăn', ''))

#     print(
#         remove_html(r'<span class="bq"><span class="br" style="height: 16px; \
#             width: 16px; font-size: 16px; background-image: \
#             url(&quot;https://static.xx.fbcdn.net/images/emoji.php/v9/te/1/16/\
#             1f622.png&quot;)">😢</span></span> hok biết ly thuỷ tinh có về lại\
#             hok <a href="/tran.pham.334839?refid=52&amp;__tn__=R">Trân Phạm\
#             </a>'))

#     print(remove_hashtag(r'#love #yêu #3000 valentine hạnh phúc'))
#     print(mask_url("Xem them tai http://www.abc.xyz"))
#     print(mask_url("Xem them tai https://www.abc.xyz"))
#     print(mask_url("Xem them tai http://abc.xyz"))
#     print(mask_url("Xem them tai https://abc.xyz"))
#     print(mask_url("Xem them tai www.abc.xyz"))

#     print(remove_repetition("Xeeeeeem điiiiiiiii", "char"))
#     print(remove_repetition("Cái xoong", "char"))
#     print(remove_repetition("yeeeeeeuuuuuuu yêêêu yêêu yêu 1000", "char"))
#     print(remove_repetition("--------------", "dash"))
