import pandas as pd
import warnings

def make_testset(category,df,dialogues,responses,cat_type):
    """
    category - High or low
    df - dataframe of the VAD
    dialogue and response - list of words - Remember I pass the list of source and target sentences after it has been tokenized and is placed in a list of list like this for example - [["Hi","my","Name","is"],["how","are",you"]]
    cat_type - Valence/Arousal/Dominance
    """
    if category.lower() == "high":
        if cat_type != "valence":
            df = df[:20]
            df['Word'] = df['Word'].str.lower()
            words = list(set(df['Word'].tolist()))
        else:
            df_poswords = df[:20]
            df_negwords = df[-5000:]
            df_poswords['Word'] = df_poswords['Word'].str.lower()
            df_negwords['Word'] = df_negwords['Word'].str.lower()
            pos_words = list(set(df_poswords['Word'].tolist()))
            pos_words.remove("honest")
            neg_words = list(set(df_negwords['Word'].tolist()))
    else:
        df = df[-20:]
        df['Word'] = df['Word'].str.lower()
        words = list(set(df['Word'].tolist()))
    
    if 'words' in locals():
        if "honest" in words:
            words.remove("honest")
    cat_type = category+ " " + cat_type
    target_questions = []
    target_responses = []
    category_type = []
    indexes = [] # need these for removing them from the original list
    for i, dialogue in enumerate(dialogues):
        switch = False
        for word in dialogue:
            if cat_type != "high valence":
                if word in words:
                    switch = True
                elif word == "honest":
                    switch = False
                    break
            else:
                if word in pos_words:
                    switch = True
                elif word =="honest":
                    switch = False
                    break
                elif switch == True and word in neg_words:
                    switch = False
                    break
        if switch == True:
            target_questions.append(' '.join(dialogues[i]))
            target_responses.append(' '.join(responses[i]))
            category_type.append(cat_type)
            indexes.append(i)
    df_new = pd.DataFrame(index=range(len(target_questions)))
    df_new['target_questions'] = pd.DataFrame(target_questions)
    df_new['target_responses'] = pd.DataFrame(target_responses)
    df_new['category_type'] = pd.DataFrame(category_type)
    df_new['indexes'] = pd.DataFrame(indexes)
    return df_new,indexes

def exclude_testset(dialogues, responses):
	"""
	dialogues - list(list(str)) of prompts
	responses - list(list(str)) of answers

	"""

	df = pd.read_excel(f'./Warriner, Kuperman, Brysbaert - 2013 BRM-ANEW expanded.xlsx')
	df = df[['Word','V.Mean.Sum','A.Mean.Sum','D.Mean.Sum']]
	df_valence = df[['Word','V.Mean.Sum']]
	df_valence = df_valence.sort_values(by=['V.Mean.Sum'],ascending=False).reset_index(drop=True)
	df_arousal = df[['Word','A.Mean.Sum']]
	df_arousal = df_arousal.sort_values(by=['A.Mean.Sum'],ascending=False).reset_index(drop=True)
	df_dominance = df[['Word','D.Mean.Sum']]
	df_dominance = df_dominance.sort_values(by=['D.Mean.Sum'],ascending=False).reset_index(drop=True)
	
	warnings.filterwarnings('ignore')
	high_valence, hv_indexes =  make_testset("high",df_valence,dialogues,responses,"valence")
	low_valence, lv_indexes =  make_testset("low",df_valence,dialogues,responses,"valence")
	#------------------------#
	high_arousal, ha_indexes =  make_testset("high",df_arousal,dialogues,responses,"arousal")
	low_arousal, la_indexes =  make_testset("low",df_arousal,dialogues,responses,"arousal")
	#------------------------#
	high_dominance, hd_indexes =  make_testset("high",df_dominance,dialogues,responses,"dominance")
	low_dominance, ld_indexes =  make_testset("low",df_dominance,dialogues,responses,"dominance")
	
	df_new = pd.concat([high_valence, low_valence, high_arousal, low_arousal, high_dominance,low_dominance]).reset_index(drop=True)
	df_new.to_excel('./test_set_removed.xlsx',index=False)

	test_indices = hv_indexes + lv_indexes + ha_indexes + la_indexes + hd_indexes + ld_indexes
	unique_test_indices = set(test_indices)
	remaining_indices = list( set(range(len(dialogues))) - unique_test_indices )
	remaining_indices.sort()


	return remaining_indices

	
