def get_model():
    pass

def get_data():
    pass


if __name__ == '__main__':

    # get data
    # sys.args
    X, ids = get_data(input_path)
    
    model = get_model()

    pred1 = model_bot.prop_predict(X)
    pred2 = model_gender.prop_predict(X)

    # ger results
    authors = {}
    for i in range(0, len(pred1)):
        if ids[i] not in authors:
            authors[ids[i]] = ([],[])
        else:
            arr1, arr2 = authors[ids[i]]
            # bot
            arr1.append(pred1[i])
            # gender
            arr2.append(pred2[i]a)
            authors[ids[i]] = (arr1, arr2)
        

    for key in authors:
        result = {}
        result['author_id'] = key
        gender_class = ['female','male']
        bot_class = ['human','bot']

        b = np.argmax(authors[key][0], axis=1)
        bot = np.argmax(np.bincount(b))

        result['bot'] = bot_class[bot]
        if bot != 1: 
            a = np.argmax(authors[key][1], axis=1)
            gender = np.argmax(np.bincount(a))
            result['gender'] = gender_class[gender]

        # write xml file
            

    # save result by author
    set_results(pred)
