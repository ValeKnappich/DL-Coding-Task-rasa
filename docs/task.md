# Task

The task is to predict the intent and entities in user utterances.

### Data Format

There is labelled training data and unlabelled test data. The score on the unlabelled test data can be checked at the course portal.

```json
{
    "0": {
        "intent": "AddToPlaylist",
        "text": "Add a tune to my elrow Guest List",
        "slots": {
            "music_item": "tune",
            "playlist_owner": "my",
            "playlist": "elrow Guest List"
        },
        "positions": {
            "music_item": [
                6,
                9
            ],
            "playlist_owner": [
                14,
                15
            ],
            "playlist": [
                17,
                32
            ]
        }
    }
}
```