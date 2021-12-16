import pytest
from os import listdir
from os.path import isdir, isfile, dirname, join
from IOKR.model.utils import SGD, MyDataset, create_path_that_doesnt_exist, project_root


class TestCreatePathThatDoesntExist():

    def test_path_dir_is_created(self, tmp_path):
        d1 = tmp_path / "Testdir"
        f1 = d1 / 'Check.txt'
        assert isdir(d1) is False, f'{d1} already exist'
        create_path_that_doesnt_exist(d1, "Check", "txt")
        assert isdir(d1) is True, f'{d1} has not been created'
        
class TestProjectRoot():

    def test_project_root(self):
        """Test for project_root function"""
        #Get the actual path
        actual_path = dirname(__file__)
        #Get the supposed root_path to check
        root_path = dirname(project_root())
        #Get the difference in length of both
        a = len(actual_path)
        b = len(root_path)
        dif_length = a - b
        #Subtract the difference to the actual path
        root_actual_path = actual_path[:-dif_length]
        #Assert that the root path from the function is the same as the root of the actual path
        assert root_actual_path == root_path, \
            f'Root_path should be {root_path}, but instead returned {root_actual_path}'
