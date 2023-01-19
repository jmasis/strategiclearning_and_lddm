import os
import pymworks

def get_animals_and_their_session_filenames(filepath):
    '''
    Returns a dict with animal names as keys (it gets their names from the
        folder names in 'input' folder--each animal should have its own
        folder with .mwk session files) and a list of .mwk filename strings as
        values.
            e.g. {'V1': ['V1_140501.mwk', 'V1_140502.mwk']}

    :param path: a string of the directory name containing animals' folders
    '''
    #TODO maybe make this better, it's slow as hell and ugly
    result = {}
    dirs_list = [each for each in os.walk(filepath)]
    for each in dirs_list[1:]:
        files_list = each[2]
        animal_name = each[0].split("/")[len(each[0].split("/")) - 1]
        result[animal_name] = [] #list of filenames
        for filename in files_list:
            if not filename.startswith('.'): #dont want hidden files
                result[animal_name].append(filename)
    
    animals_and_their_session_filenames = result
    return animals_and_their_session_filenames

def get_trials_from_all_sessions(animal_name, sessions, filepath):
    '''
    Returns a list with all trials from all sessions for an animal
        e.g. [{'trial_num': 1, ...}, ..., {'trial_num':345 ...}, ..., {'trial_num': 1, ...} ... ]
    '''
    print "Starting analysis for ", animal_name
    all_trials_all_sessions = []
    for session in sessions:
        trials = get_session_trials(animal_name, session, filepath)
        all_trials_all_sessions += trials
    return all_trials_all_sessions

def group_trials_from_each_session(animal_name, sessions, filepath):
    '''
    Returns a dict with sessions as keys and trials from that session as values
        e.g. {'AJ1_170613.mwk': 

            [{'trial_num': 1, 'behavior_outcome': 'failure','stm_size': 40.0,},
            {'trial_num': 2,'behavior_outcome': 'success','stm_size': 35.0} ... ]

              'AJ1_170614.mwk': [...]
             }
    '''
    print "Starting analysis for ", animal_name
    trials_from_each_session = {}
    for session in sessions:
        trials = get_session_trials(animal_name, session, filepath)
        key = session
        trials_from_each_session[key] = trials
    return trials_from_each_session

def get_session_trials(animal_name, session_filename, filepath):
    '''
    Returns a time-ordered list of dicts, where each dict is info about a trial.
    e.g. [{'trial_num': 1,
           'behavior_outcome': 'failure',
           'stm_size': 40.0,
           },
          {'trial_num': 2,
           'behavior_outcome': 'success',
           'stm_size': 35.0
           }]

    :param animal_name: name of the animal string
    :param session_filename: filename for the session (string)
    '''

    #TODO: unfuck this: hard coded paths not ideal for code reuse

    print 'session_filename'
    print session_filename 

    path = filepath + animal_name + '/' + session_filename
    #path = 'input/' + 'fullmatrixnewlighting/' + animal_name + '/' + session_filename

    df = pymworks.open_file(path)
    events = df.get_events([
        'Announce_TrialStart',#when value = 1, used for identifying real trials (some information will be before and some after this variable)
        'Announce_TrialEnd',#when value = 1, used for identifying the end of real trials
        '#stimDisplayUpdate',#time display is updated with stimulus, blank screen, etc on screen
        'Announce_AcquirePort1',#time when first lick is registered as response
        #'LickInput2', #center port response
        'Announce_AcquirePort3',#time when first lick is registered as response
        'success',#behavioral outcome
        'failure',#behavioral outcome
        'ignore',#behavioral outcome
        'BlobIdentityIdx'#index assigned to blob 1 or 2 to identify them
        ]
    )

    trials = []
    trial_num = 1
    for index, event in enumerate(events):
        if (event.name == "Announce_TrialStart" and
        event.value == 1):
            trial = {
                'trial_num': trial_num,
                'stim_presentation_time': None,
                'behavior_outcome_time': None,
                'reaction_time': None,
                'behavior_outcome': None,
                'BlobIdentityIdx': None,
            }
            
            #COLLECT STIMULUS PROPERTIES FOR THIS TRIAL
            stm_properties = [
                'BlobIdentityIdx']
            for stm_property in stm_properties:
                try:
                    if events[index - 1].name == stm_property:
                        trial[stm_property] = events[index - 1].value
                except IndexError:
                    print stm_property + ' out of range for session', session_filename, index

            #COLLECT BEHAVIORAL OUTCOME FOR THIS TRIAL AND ITS TIME
            #Because the animal can respond after the stimulus disappears and there's a
            #stimulus update after that, we need to look for these in two different spots
            #hence the if and elif statements
            try:
                if events[index + 1].name == '#stimDisplayUpdate':
                    trial['stim_presentation_time'] = events[index + 1].time
                #if there is a second #stimDisplayUpdate it's a blank screen and 
                #it means the animal answered the trial after the stimulus had disappeared
                #so no need to save it
            except IndexError:
                print 'stim_presentation_time out of range for session', session_filename, index
            
            try:
                if events[index + 2].name in ['Announce_AcquirePort1', 'Announce_AcquirePort3']:
                    trial['behavior_outcome_time'] = events[index + 2].time
                elif events[index + 3].name in ['Announce_AcquirePort1', 'Announce_AcquirePort3']:
                    trial['behavior_outcome_time'] = events[index + 3].time
            except IndexError:
                print 'behavior_outcome_time out of range for session', session_filename, index
            
            try:
                if events[index + 3].name in ['success', 'failure', 'ignore']:
                    trial['behavior_outcome'] = events[index + 3].name
                elif events[index + 4].name in ['success', 'failure', 'ignore']:
                    trial['behavior_outcome'] = events[index + 4].name
            except IndexError:
                print 'behavior_outcome out of range for session', session_filename, index


            if trial['behavior_outcome_time'] is not None and trial['stim_presentation_time'] is not None:
                trial['reaction_time'] = (trial['behavior_outcome_time'] - trial['stim_presentation_time'])*.001


            #print trial
            
            #CHECK OUTCOME AND ESSENTIAL STM PROPERTIES HAVE BEEN COLLECTED FOR THIS TRIAL
            #(not including all stm properties so the code is flexible for simpler protocols)
            if (
            trial['behavior_outcome'] is not None and
            trial['stim_presentation_time'] is not None and
            trial['behavior_outcome_time'] is not None and
            trial['BlobIdentityIdx'] is not None):
                trials.append(trial)
                trial_num += 1
    #print trials
    print 'number of trials'
    print len(trials)

    return trials
