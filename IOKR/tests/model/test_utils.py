import pytest
from os import listdir
from os.path import isdir, isfile
from IOKR.model.utils import SGD, MyDataset, create_path_that_doesnt_exist, project_root


class TestCreatePathThatDoesntExist():

    def test_path_dir_is_created(self, tmp_path):
        d1 = tmp_path / "Testdir"
        f1 = d1 / 'Check.txt'
        assert isdir(d1) is False, f'{d1} already exist'
        create_path_that_doesnt_exist(d1, "Check", "txt")
        assert isdir(d1) is True, f'{d1} has not been created'

#class TestProjectRoot():

#    def test_project_root(self, tmp_path):
#        d1 = tmp_path / 'd1'
#        d2 = d1 / 'd2'
#        d3 = d2 / 'd3'
#        d4 = d3 / 'd4'
