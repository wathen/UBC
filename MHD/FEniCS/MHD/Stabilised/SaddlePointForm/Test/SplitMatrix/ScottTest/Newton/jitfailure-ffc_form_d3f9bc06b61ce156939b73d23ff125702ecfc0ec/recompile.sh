#!/bin/bash
# Execute this file to recompile locally
c++ -Wall -shared -fPIC -std=c++11 -O2 -I/Users/michaelwathen/anaconda/lib/python2.7/site-packages/ffc/backends/ufc -I/Users/michaelwathen/.cache/dijitso/include ffc_form_d3f9bc06b61ce156939b73d23ff125702ecfc0ec.cpp -L/Users/michaelwathen/.cache/dijitso/lib -Wl,-rpath,/Users/michaelwathen/.cache/dijitso/lib -ldijitso-ffc_element_c8b3a924d462ab9bfe5a04f9524c8840db969905 -ldijitso-ffc_element_8fb97a81157753ea74ad04769eb1aebe604ce903 -ldijitso-ffc_element_1ca09c7da323f43cc9d05082aef65289352ff0a2 -Wl,-install_name,/Users/michaelwathen/.cache/dijitso/lib/libdijitso-ffc_form_d3f9bc06b61ce156939b73d23ff125702ecfc0ec.so -olibdijitso-ffc_form_d3f9bc06b61ce156939b73d23ff125702ecfc0ec.so