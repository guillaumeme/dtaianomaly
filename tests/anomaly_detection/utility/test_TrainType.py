
import pytest
from dtaianomaly.anomaly_detection.utility.TrainType import TrainType


class TestTrainType:

    def test_unsupervised__str__(self):
        assert TrainType.UNSUPERVISED.__str__() == 'unsupervised'

    def test_unsupervised_can_solve_train_type_data(self):
        assert TrainType.UNSUPERVISED.can_solve_train_type_data('unsupervised')
        assert TrainType.UNSUPERVISED.can_solve_train_type_data('semi-supervised')
        assert TrainType.UNSUPERVISED.can_solve_train_type_data('supervised')

    def test_semi_supervised__str__(self):
        assert TrainType.SEMI_SUPERVISED.__str__() == 'semi-supervised'

    def test_semi_supervised_can_solve_train_type_data(self):
        assert not TrainType.SEMI_SUPERVISED.can_solve_train_type_data('unsupervised')
        assert TrainType.SEMI_SUPERVISED.can_solve_train_type_data('semi-supervised')
        assert TrainType.SEMI_SUPERVISED.can_solve_train_type_data('supervised')

    def test_supervised__str__(self):
        assert TrainType.SUPERVISED.__str__() == 'supervised'

    def test_supervised_can_solve_train_type_data(self):
        assert not TrainType.SUPERVISED.can_solve_train_type_data('unsupervised')
        assert not TrainType.SUPERVISED.can_solve_train_type_data('semi-supervised')
        assert TrainType.SUPERVISED.can_solve_train_type_data('supervised')

    def test_can_solve_train_type_data_invalid(self):
        with pytest.raises(ValueError):
            TrainType.UNSUPERVISED.can_solve_train_type_data('some random type')
        with pytest.raises(ValueError):
            TrainType.SUPERVISED.can_solve_train_type_data('some random type')
        with pytest.raises(ValueError):
            TrainType.SUPERVISED.can_solve_train_type_data('some random type')
