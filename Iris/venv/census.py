import pandas
import KNNClassifier


def run(url, header, categories=None, Nan="?", k=3):
    data = read(url, header)
    #print data
    if categories is not None:
        for key in categories:
            conversion = categorize(data[key], categories[key], Nan=Nan)
            #print conversion.keys()
            data[key] = data[key].map(conversion)
            #for x in xrange(len(data[key])):
            #    #print data[key][x]
            #    if data[key][x] in conversion.keys():
            #        print "converting..."
            #        data[key][x] = conversion[data[key][x]]
    else:
        for head in header:
            if numpy.issubdtype(data[head].dtype, numpy.number):
                conversion = categorize(data[head], build_options_list(data[head], Nan))
                data[head] = data[head].map(conversion)
    print data[header[13]][14]
    #data.replace(Nan, "0")
    #for key in header:
    #    #print data[key]
    #    for y in xrange(len(data[key])):
    #        if data[key][y] == "?":
    #            print "Guessing..."
    #            data[key][y] = 0.0
    print data
    print data[header[13]][14]
    data.apply(pandas.to_numeric)
    print data[header[13]][14]
    data = data.as_matrix()
    print data[14][13]
    print data
    print "data tested with K = ", k, ": ", KNNClassifier.test_classifier(data[:, range(13)], data[:, 14], k)


def read(url, header):
    data = pandas.read_csv(url, names= header, skipinitialspace= True)
    return data


def categorize(data, options, Nan="?"):
    dictionary = {}
    distribution = 4.0 / len(options)
    offset = distribution / 2

    # normalizing possible options to float values between -2 and 2
    for x in xrange(len(options)):
        dictionary.update({options[x]: ((distribution * x) + offset - 2)})

    dictionary.update({Nan: 0})

    return dictionary


def build_options_list(list, Nan=None):
    result = []
    for x in list:
        if x not in result.keys() and x != Nan:
            result.append(x)

    return result



if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    header = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
              "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
              "50K-threshold"]
    categories = {"workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "State-gov", "Local-gov",
                                "Without-pay", "Never-worked"]}
    categories.update({"education": ["Doctorate", "Masters", "Prof-school", "Bachelors", "Assoc-acdm", "Assoc-voc",
                                     "Some-college", "HS-grad", "12th", "11th", "10th", "9th", "7th-8th", "5th-6th",
                                     "1st-4th", "Preschool"]})
    categories.update({"marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                                          "Married-spouse-absent", "Married-AF-spouse"]})
    categories.update({"occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                                      "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                                      "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                                      "Armed-Forces"]})
    categories.update(
        {"relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]})
    categories.update({"race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]})
    categories.update({"sex": ["Female", "Male"]})
    categories.update({"native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                                          "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
                                          "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
                                          "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
                                          "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
                                          "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
                                          "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]})
    categories.update({"50K-threshold": ["<=50K", ">50K"]})
    run(url, header, categories=categories)