import re


def clean(comment):
    # Convert to lower case , so that Hi and hi are the same
    comment = comment.lower()
    # remove \n
    comment = re.sub("\\n", "", comment)
    # # remove leaky elements like ip,user
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", comment)
    # # removing usernames
    comment = re.sub("\[.*\]", "", comment)

    return comment
