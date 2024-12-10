### transformer, note:
following hkproj Umar Jamil's tutorial, trying to understand full inner working by coding along

### for the future:
- <ins> will convert code to c++ and hyperoptimize (the hard part) </ins>
- want to try and NOT use libraries for cpp model, Libtorch is currently being used simply due to the amount of energy i have to build low level things
- will also use bbtorch once that is fully implemented
- see diagram if you want, also on my portfolio git pages site

## just clone and run if you want, i suggest using debugger so you can pause
- feel free to edit and pull req if you want! can be filling in comments or writting cpp code
- i havent written tests though
- <ins>NOTE:</ins> currently not setup to retrain on prev. weights! if you run this once, it will make a weights folder, but we need to go into config.py as well as the train_model() function to add this. ill probably do this soon.

### links:
portfolio w/ diagram: https://grahamstelzer.github.io/#
EXTREMELY helpful visuals: https://github.com/hkproj/
umar jamils actual working: https://github.com/hkproj/pytorch-transformer,
note, definitely several steps ahead of this implementation, notably uses beam search. i want to take what we currently have and turn it into cpp


