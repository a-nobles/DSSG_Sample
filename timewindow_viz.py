#!/usr/bin/env python


import click
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta


def read_episodes_data(filename, episode_type, begin_day, end_day):
    """Read and clean up the episodes data"""
    episodes_data = pd.read_csv(filename, encoding='utf-8')
    # Retain only needed columns
    episodes_data = episodes_data.loc[:, ['ID', 'episode', 'EndDate',
                                      'Certainty', 'episodeNum']]
    # Rename columns for consistency
    episodes_data = episodes_data.rename(columns={"ID": "participant",
                                                  "episode": "episode_type",
                                                  "EndDate": "episode_date",
                                                  "Certainty": "certainty",
                                                  "episodeNum": "episode_num"})
    # Select only episodes that align with the episode_type
    episodes_data = episodes_data[episodes_data.episode_type == episode_type]
    # Concatenate the participant and episode columns
    episodes_data['participant_episode'] = (episodes_data['participant'] +
                                            "_" + episodes_data['episode_num'])
    # Convert date to a datetime
    dt_converter = lambda x: datetime.strptime(x, '%m/%d/%y')
    episodes_data['episode_date'] = episodes_data['episode_date'].apply(dt_converter)
    # Assign a begin_date and end_date for each time window
    episodes_data['begin_date'] = (episodes_data['episode_date'] -
                                   timedelta(days=begin_day))
    episodes_data['end_date'] = (episodes_data['episode_date'] +
                                 timedelta(days=end_day))
    episodes_data.to_csv('relabeled_episodes.csv', encoding='utf-8',
                         index=False)
    return episodes_data


def read_sms_data(filename, episode_type):
    """Read and clean up the SMS data."""
    sms_data = pd.read_csv(filename, encoding='utf-8')
    # Select only outgoing messages
    sms_data = sms_data[sms_data.in_out == 'out']
    # Retain only needed columns
    sms_data = sms_data.loc[:, ['participant', 'date', 'time']]
    # Create a comparison date for labeling purposes
    sms_data['comparison_date'] = [datetime.strptime(x, '%Y-%m-%d')
                                   for x in sms_data['date']]
    # del sms_data['comparison_date']
    return sms_data


def join_episode_data(sms_data, episodes_data):
    """Merges sms_data and episodes_data based on the participant"""
    print(sms_data.dtypes)
    print(episodes_data.dtypes)
    # merge the two datasets together based on the values of participant
    data = sms_data.merge(episodes_data, how='left', on=('participant',))

    # filter the datasets based on whether the communication date falls within
    # the episode's day range
    data = data[(data['comparison_date'] >= data['begin_date']) &
                (data['comparison_date'] <= data['end_date'])]

    # set the relative_date column
    data['relative_date'] = data['comparison_date'] - data['episode_date']
    data['relative_date'] = data['relative_date'].dt.days
    data.to_csv('trans_data.csv', encoding='utf-8', index=False)
    return data


def pivot_dump(sms_data):
    """Pivot on 'relative_date' and add columns for participants."""
    grouped = sms_data.groupby(['participant_episode', 'relative_date',
                                'certainty'])
    counts = grouped.count()
    deindexed = counts.reset_index()
    # before saving out, cut out some columns, and put in count
    deindexed.to_csv('deindexed.csv', encoding='utf-8', index=False)
    pivot = deindexed.pivot(index='relative_date',
                            columns='participant_episode',
                            values='episode_type')
    pivot.to_csv('pivoted_data.csv', encoding='utf-8', index=True)


def aggregate(sms_data):
    """Aggregate on mean, stddev, by relative_date."""
    grouped = sms_data.groupby(['participant_episode', 'relative_date'])
    counts = grouped.count()
    deindexed = counts.reset_index()

    by_date = deindexed.groupby('relative_date')
    aggregates = by_date.aggregate([np.mean, np.std])

    reset = aggregates.reset_index()
    cols = reset.columns.droplevel(0).tolist()
    cols[0] = 'relative_date'
    reset.columns = cols
    # print(reset.head())
    reset.to_csv('reset_data.csv', encoding='utf-8', index=False)
    return reset


def norm_episode_type(type_code):
    """This normalizes episode type codes to their standard forms. Returns None
    if the type code is invalid."""
    types = {
        'a': 'Attempt',
        'i': 'Ideation',
        'd': 'Depression',
        'p': 'Positive',
        }
    try:
        return types[type_code[0].lower()]
    except:
        raise Exception('Invalid episode type code: ' + type_code)


@click.command()
@click.option('--input-file', '-i',
              help='Input aggregate data file with all participants.')
@click.option('--episodes-file', '-E',
              help='Input aggregate episodes file with participants.')
@click.option('--begin-day', '-b', type=int,
              help='The number of days before an episode to begin graphing '
                   'interactions.')
@click.option('--end-day', '-e', type=int,
              help='The number of days after an episode to stop graphing '
                   'interactions.')
@click.option('--episode-type', '-t', default='a', type=norm_episode_type,
              help='The type of episode being graphed. One of [a]ttempt, '
                   ' [i]deation, [d]epression, or [p]ositive. '
                   'Default is attempt.')
@click.option('--dump-participants', '-d', default=False,
              help='Dump out the pre-aggregated data for each participant '
                   'to a file named "daily-sms-count.csv".')
def main(input_file, episodes_file, begin_day, end_day, episode_type,
         dump_participants):
    episodes_data = read_episodes_data(episodes_file, episode_type, begin_day,
                                       end_day)
    sms_data = read_sms_data(input_file, episode_type)
    sms_data = join_episode_data(sms_data, episodes_data)
    if dump_participants:
        pivot_dump(sms_data.copy())
    aggregate(sms_data)

    # TODO generate viz
    # TODO: also dump out a graph for each participant.


if __name__ == '__main__':
    main()
