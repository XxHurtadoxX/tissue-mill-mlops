"""Archivo temporal con un error deliberado, para verificar que la protección
de rama bloquea la fusión. Se borra en cuanto se compruebe."""
import os
import sys


def funcion_con_error( ):
    variable_sin_usar = 42
    return   "espaciado incorrecto a proposito"
