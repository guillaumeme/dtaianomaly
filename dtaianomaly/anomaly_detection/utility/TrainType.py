
import enum


class TrainType(enum.Enum):
    SUPERVISED = 1
    SEMI_SUPERVISED = 2
    UNSUPERVISED = 3

    def can_solve_train_type_data(self, train_type_data: str) -> bool:

        # Check the input
        if train_type_data not in ['supervised', 'semi-supervised', 'unsupervised']:
            raise ValueError(f"Unknown train type '{train_type_data}'!"
                             f"Valid train types are 'supervised', 'semi-supervised' and 'unsupervised'!")

        # If the train type of this is unsupervised, then it can solve all datasets.
        if self == TrainType.UNSUPERVISED:
            return True

        # If the train type of this is semi-supervised, then it can solve both supervised and semi-supervised datasets.
        if self == TrainType.SEMI_SUPERVISED:
            return train_type_data == 'semi-supervised' or train_type_data == 'supervised'

        # If the train type of this is supervised, then it can only solve supervised datasets.
        if self == TrainType.SUPERVISED:
            return train_type_data == 'supervised'

    def __str__(self):
        return self.name.lower().replace('_', '-')
