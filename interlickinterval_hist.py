from __future__ import division
import os
import numpy as np
import pymworks
import matplotlib.pyplot as plt
from collections import Counter
from mw_parse_initial_170623 import get_animals_and_their_session_filenames


def analyze_sessions(filepath):
    animals_and_their_session_filenames = get_animals_and_their_session_filenames(filepath)
    
    for animal, sessions in animals_and_their_session_filenames.iteritems():
    	get_data_for_figure(animal, sessions, filepath)

def get_data_for_figure(animal_name, sessions, filepath):

    licktimes_merged_dict = {}
    all_lick_times_by_session = []

    for session in sessions:
        path = filepath + animal_name + '/' + session
        lick_times = lickport_times(path)
        all_lick_times_by_session.append(lick_times)

        # merge dicts
        for key in lick_times.iterkeys():
            try:
                licktimes_merged_dict[key].extend(lick_times[key])
            except KeyError:
                licktimes_merged_dict[key] = lick_times[key]

    hist_per_animal(animal_name, sessions, licktimes_merged_dict)

    print 'made histogram for', animal_name


def hist_per_animal(animal_name, sessions, licktimes_merged_dict):
	# get animal name
	animal = animal_name

	# makes list of just dates as session indices for plot filename
	session_indices = []
	for i in range(len(sessions)):
		index = sessions[i].split('_')[1].split('.')[0]
		session_indices.append(index)
		session_indices = sorted(session_indices)

	# plt.hist parameters
	bins = 50
	range_ = (0,400)
	log = False
	alpha = 0.2

	# data
	x1 = licktimes_merged_dict['lick_2_3']
	x2 = licktimes_merged_dict['lick_2_1']
	x3 = licktimes_merged_dict['lick_1_2']
	x4 = licktimes_merged_dict['lick_3_2']

	# labels
	l1 = 'lick_2_3'
	l2 = 'lick_2_1'
	l3 = 'lick_1_2'
	l4 = 'lick_3_2'

	# set weights to be equal in order to have the plot normalize correctly and not to the integral involving the bar width
	w1 = np.ones_like(x1)/float(len(x1))
	w2 = np.ones_like(x2)/float(len(x2))
	w3 = np.ones_like(x3)/float(len(x3))
	w4 = np.ones_like(x4)/float(len(x4))

	# plotting:
	#to plot just total counts, comment out weights and normed parameters
	#to plot with correct normalization include weights and not normed
	#to plot with the traditional normalization that takes into account bar width when integrating use normed = True
	hist1 = plt.hist(x1, 
		range = range_, 
		bins = bins, 
		log = log, 
		alpha = alpha, 
		label = l1)
	hist2 = plt.hist(x2, 
		range = range_, 
		bins = bins, 
		log = log, 
		alpha = alpha, 
		label = l2)
	hist3 = plt.hist(x3, 
		range = range_, 
		bins = bins, 
		log = log, 
		alpha = alpha, 
		label = l3)
	hist4 = plt.hist(x4, 
		range = range_, 
		bins = bins, 
		log = log, 
		alpha = alpha, 
		label = l4)

	#create legend by hand because must do so for superimposed histograms
	plt.legend()
	#plot labels
	plt.xlabel('time difference between licks (ms)')
	plt.ylabel('count')
	plt.title('%s full cross first 10 sessions' % animal)
	plt.savefig('%s_timediffbtwnlicks_merged_fullcross_first10_%s_to_%s.pdf' %(animal, session_indices[0], session_indices[len(session_indices)-1]), bbox_inches = 'tight')
	plt.close()
	#plt.show()


def lickport_times(path):
    df = pymworks.open_file(path)

    lickevents = df.get_events([
            'LickInput1',
            'LickInput2',
            'LickInput3'
        ])
    
    '''
    trying to capture lick times and switch times between ports
    '''
    
    lick_all = []
    lick_1 = []; lick_1_1 = []; lick_1_2 = []; lick_1_3 = [];
    lick_2 = []; lick_2_1 = []; lick_2_2 = []; lick_2_3 = [];
    lick_3 = []; lick_3_1 = []; lick_3_2 = []; lick_3_3 = [];
    
    
    thresh = 5 # lickport value that separates "off" versus "on" (< thresh is off, > thresh is on) 
    update_time = 8 # ms that it takes the capacitive sensor to switch from one value to another
        
    for i in range(len(lickevents)): # go through all logged lick events for a session
        if i >= (len(lickevents) - 1) or i == 0: # loop goes through i+1 & i-1, so have to terminate when i = len(lickevents) - 1 or when i = 0
            pass
        else:
            if lickevents[i].name == 'LickInput1':
                if lickevents[i].value < thresh: # when the value is an "off" value, it can be a switch between ports or another lick on same port
                    if lickevents[i-1].name == 'LickInput1' and lickevents[i-1].value > thresh: # confirm it is an off value by checking for an "on" value 8 ms before
                        if round((lickevents[i].time - lickevents[i-1].time)*0.001) == update_time:
                            if lickevents[i+1].name == 'LickInput1' and lickevents[i+1].value > thresh:
                                lick_1_1.append((lickevents[i+1].time - lickevents[i-1].time)*0.001) # time difference between licks is set by lick before "off" value and the following "on" value
                            elif lickevents[i+1].name == 'LickInput2' and lickevents[i+1].value > thresh:
                                lick_1_2.append((lickevents[i+1].time - lickevents[i-1].time)*0.001)
                            elif lickevents[i+1].name == 'LickInput3' and lickevents[i+1].value > thresh:
                                lick_1_3.append((lickevents[i+1].time - lickevents[i-1].time)*0.001)
                        else: # value = 0 should always happen 8 ms after a higher value on the same lickport
                            pass
                    else: # if (lickevents[i-1].name == 'LickInput1' and lickevents[i-1].value > 0) is false then it's not a real lick b/c value = 0 should always follow a lick on the same lickport
                        pass               
                elif lickevents[i].value > thresh: # when the value is an "on" value, the next lick can only be a lick on the same port (otherwise touching both simultaneously or something like that)
                    if lickevents[i-1].name == 'LickInput1':
                        if lickevents[i-1].value < thresh: # this lick will have been counted by above logic
                            pass
                        elif lickevents[i-1].value > thresh:
                            if round((lickevents[i].time - lickevents[i-1].time)*0.001) == update_time: # check if this lick is a value update (8 ms) or a real lick
                                if lickevents[i+1].name == 'LickInput1' and lickevents[i+1].value > thresh:
                                    lick_1_1.append((lickevents[i+1].time - lickevents[i-1].time)*0.001) # use time of i-1 and i+1
                                else: # two licks above thresh can only be on the same lickport to be real (there needs to be a value below thresh to count as a real switch between ports)
                                    pass
                            elif round((lickevents[i].time - lickevents[i-1].time)*0.001) > update_time:
                                if lickevents[i+1].name == 'LickInput1' and lickevents[i+1].value > thresh:
                                    if round((lickevents[i+1].time - lickevents[i].time)*0.001) > update_time:
                                        lick_1_1.append((lickevents[i+1].time - lickevents[i].time)*0.001) # use time of i and i+1
                                    else: # if this is a lickport update (8 ms timing difference) for the next, then catch the lick on the next round
                                        pass
                                else: # two licks above thresh can only be on the same lickport to be real (there needs to be a value below thresh to count as a real switch between ports)
                                    pass
                    else: # if value > 0 and last lick was on a different lickport, then not a real lick
                        pass

            elif lickevents[i].name == 'LickInput2':
                if lickevents[i].value < thresh:
                    if lickevents[i-1].name == 'LickInput2' and lickevents[i-1].value > thresh:
                        if round((lickevents[i].time - lickevents[i-1].time)*0.001) == update_time:
                            if lickevents[i+1].name == 'LickInput1' and lickevents[i+1].value > thresh:
                                lick_2_1.append((lickevents[i+1].time - lickevents[i-1].time)*0.001)
                            elif lickevents[i+1].name == 'LickInput2' and lickevents[i+1].value > thresh:
                                lick_2_2.append((lickevents[i+1].time - lickevents[i-1].time)*0.001)
                            elif lickevents[i+1].name == 'LickInput3' and lickevents[i+1].value > thresh:
                                lick_2_3.append((lickevents[i+1].time - lickevents[i-1].time)*0.001)
                        else: # value = 0 should always happen 8 ms after a higher value on the same lickport
                            pass
                    else: # if (lickevents[i-1].name == 'LickInput1' and lickevents[i-1].value > 0) is false then it's not a real lick b/c value = 0 should always follow a lick on the same lickport
                        pass               
                elif lickevents[i].value > thresh:
                    if lickevents[i-1].name == 'LickInput2':
                        if lickevents[i-1].value < thresh: # this lick will have been counted by above logic
                            pass
                        elif lickevents[i-1].value > thresh:
                            if round((lickevents[i].time - lickevents[i-1].time)*0.001) == update_time: # check if this lick is a value update (8 ms) or a real lick
                                if lickevents[i+1].name == 'LickInput2' and lickevents[i+1].value > thresh:
                                    lick_2_2.append((lickevents[i+1].time - lickevents[i-1].time)*0.001) # use time of i-1 and i+1
                                else: # two licks above thresh can only be on the same lickport to be real (there needs to be a value below thresh to count as a real switch between ports)
                                    pass
                            elif round((lickevents[i].time - lickevents[i-1].time)*0.001) > update_time:
                                if lickevents[i+1].name == 'LickInput2' and lickevents[i+1].value > thresh:
                                    if round((lickevents[i+1].time - lickevents[i].time)*0.001) > update_time:
                                        lick_2_2.append((lickevents[i+1].time - lickevents[i].time)*0.001) # use time of i and i+1
                                    else: # if this is a lickport update (8 ms timing difference) for the next, then catch the lick on the next round
                                        pass
                                else: # two licks above thresh can only be on the same lickport to be real (there needs to be a value below thresh to count as a real switch between ports)
                                    pass
                    else: # if value > 0 and last lick was on a different lickport, then not a real lick
                        pass

            elif lickevents[i].name == 'LickInput3':
                if lickevents[i].value < thresh:
                    if lickevents[i-1].name == 'LickInput3' and lickevents[i-1].value > thresh:
                        if round((lickevents[i].time - lickevents[i-1].time)*0.001) == update_time:
                            if lickevents[i+1].name == 'LickInput1' and lickevents[i+1].value > thresh:
                                lick_3_1.append((lickevents[i+1].time - lickevents[i-1].time)*0.001)
                            elif lickevents[i+1].name == 'LickInput2' and lickevents[i+1].value > thresh:
                                lick_3_2.append((lickevents[i+1].time - lickevents[i-1].time)*0.001)
                            elif lickevents[i+1].name == 'LickInput3' and lickevents[i+1].value > thresh:
                                lick_3_3.append((lickevents[i+1].time - lickevents[i-1].time)*0.001)
                        else: # value = 0 should always happen 8 ms after a higher value on the same lickport
                            pass
                    else: # if (lickevents[i-1].name == 'LickInput1' and lickevents[i-1].value > 0) is false then it's not a real lick b/c value = 0 should always follow a lick on the same lickport
                        pass               
                elif lickevents[i].value > thresh:
                    if lickevents[i-1].name == 'LickInput3':
                        if lickevents[i-1].value < thresh: # this lick will have been counted by above logic
                            pass
                        elif lickevents[i-1].value > thresh:
                            if round((lickevents[i].time - lickevents[i-1].time)*0.001) == update_time: # check if this lick is a value update (8 ms) or a real lick
                                if lickevents[i+1].name == 'LickInput3' and lickevents[i+1].value > thresh:
                                    lick_3_3.append((lickevents[i+1].time - lickevents[i-1].time)*0.001) # use time of i-1 and i+1
                                else: # two licks above thresh can only be on the same lickport to be real (there needs to be a value below thresh to count as a real switch between ports)
                                    pass
                            elif round((lickevents[i].time - lickevents[i-1].time)*0.001) > update_time:
                                if lickevents[i+1].name == 'LickInput3' and lickevents[i+1].value > thresh:
                                    if round((lickevents[i+1].time - lickevents[i].time)*0.001) > update_time:
                                        lick_3_3.append((lickevents[i+1].time - lickevents[i].time)*0.001) # use time of i and i+1
                                    else: # if this is a lickport update (8 ms timing difference) for the next, then catch the lick on the next round
                                        pass
                                else: # two licks above thresh can only be on the same lickport to be real (there needs to be a value below thresh to count as a real switch between ports)
                                    pass
                    else: # if value > 0 and last lick was on a different lickport, then not a real lick
                        pass
                    
    lick_times = {
        # 'lick_all': lick_all,
        # 'lick_1': lick_1,
        'lick_1_1': lick_1_1,
        'lick_1_2': lick_1_2,
        'lick_1_3': lick_1_3,
        # 'lick_2': lick_2,
        'lick_2_1': lick_2_1,
        'lick_2_2': lick_2_2,
        'lick_2_3': lick_2_3,
        # 'lick_3': lick_3,
        'lick_3_1': lick_3_1,
        'lick_3_2': lick_3_2,
        'lick_3_3': lick_3_3
        }
        
    return lick_times

if __name__ == "__main__":
    filepath = '_temp/'
    analyze_sessions(filepath)

