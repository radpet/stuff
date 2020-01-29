from util import load_clean_train, ys, TEXT


def run():
    train = load_clean_train()
    print('Train shape', train.shape)

    for y in ys:
        print('Label', y, 'sample size is', train[y].sum())
        print(train[ train[y] == 1 ].sample(5)[TEXT].values)
        print()


if __name__ == '__main__':
    run()
