LOG = True


# A small logging function that can be disabled by setting LOG to False
def info(message):
    if LOG:
        print(message)
