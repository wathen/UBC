#!/bin/bash
# Execute this file to recompile locally
c++ -Wall -shared -fPIC -std=c++11 -O2 -I/Users/michaelwathen/anaconda/lib/python2.7/site-packages/ffc/backends/ufc -I/Users/michaelwathen/.cache/dijitso/include ffc_form_81eb204e861bf1e571ab148888289af245a7a254.cpp -L/Users/michaelwathen/.cache/dijitso/lib -Wl,-rpath,/Users/michaelwathen/.cache/dijitso/lib -ldijitso-ffc_element_c8b3a924d462ab9bfe5a04f9524c8840db969905 -ldijitso-ffc_element_8fb97a81157753ea74ad04769eb1aebe604ce903 -ldijitso-ffc_element_de405066b06fb2b5ba0b8c46bdd421a8e598a2d1 -ldijitso-ffc_element_79df4a7e74199bc8f1574fb4e4728cf962fe6307 -ldijitso-ffc_element_1ca09c7da323f43cc9d05082aef65289352ff0a2 -ldijitso-ffc_element_0a08f213f0f7814b84ceffd6cb8a4eb644bea3d7 -Wl,-install_name,/Users/michaelwathen/.cache/dijitso/lib/libdijitso-ffc_form_81eb204e861bf1e571ab148888289af245a7a254.so -olibdijitso-ffc_form_81eb204e861bf1e571ab148888289af245a7a254.so