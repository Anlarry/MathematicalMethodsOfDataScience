import os, json

def has_file(File:str):
    def docator(func):
        def f(*args, **kargs):
            # vec_set_file = doc + '_vec_set.json'
            if os.path.exists(File):
                with open(File) as F:
                    return json.load(F)
                # return json.load(F)
            else :
                res = func(*args, **kargs)
                with open(File, 'w') as F:
                    try:
                        json.dump(res, F)
                    except TypeError:
                        try:
                            json.dump(list(res), F)                        
                        except TypeError:
                            json.dump([list(x) for x in res], F)
                return res
        return f
    return docator