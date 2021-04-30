from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib import pyplot as plt
from matplotlib import image
from matplotlib import backend_bases as bb
from mpl_toolkits import mplot3d
import concurrent.futures
import numpy as np
import pandas as pd
import os
import time
from backend import dzt_func, dzt_filters
from popupWindows import Export_Dialog, Alert_Dialog, Writing_Dialog


#----------------------------------------------------------#
class help_tab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(help_tab, self).__init__()
        self.layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("GPR Data Visualizer")
        title.setStyleSheet("font: bold 30px;")
        title.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.layout.addWidget(title)

        sub_title = QtWidgets.QLabel("A .DZT conversion and analysis tool sponsored by the United States Army Corps of Engineers.")
        sub_title.setStyleSheet("font: bold 15px;")
        sub_title.setAlignment(QtCore.Qt.AlignHCenter)
        self.layout.addWidget(sub_title)

        begin_line = QtWidgets.QLabel("Begin by pressing the \'+\' button to import .DZT files and open a new tab.")
        begin_line.setStyleSheet("font: bold 15px;")
        begin_line.setAlignment(QtCore.Qt.AlignHCenter)
        self.layout.addWidget(begin_line)

        doc_line = QtWidgets.QLabel("Check documentation for usage instructions.")
        doc_line.setStyleSheet("font: 15px;")
        doc_line.setAlignment(QtCore.Qt.AlignHCenter)
        self.layout.addWidget(doc_line)
#----------------------------------------------------------#
# Class instantiated to store the data and allow for user interaction
class data_tab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(data_tab, self).__init__()
        # these first 3 lines open up a file explorer for users to open a file anytime a tab is opened up, to be operated on later
        # self.import_data_box = QtWidgets.QFileDialog(self, )
        self.import_data_box = QtWidgets.QFileDialog(self)
        self.import_data_box.setNameFilters(["GPR data (*.dzt)"])
        self.import_data_box.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        self.import_data_box.setWindowTitle("Select Files to Open") 
        self.import_data_box.exec_()
        # end the file system stuff an into the actual tab stuff
        self.files_paths = [os.path._getfullpathname(f) for f in self.import_data_box.selectedFiles()]
        self.active_filters = {}
        self.filter_param_list = {
            'Horizontal background removal' : [1, 'window='],
            'Vertical triangular FIR bandpass' : [1, 'freqmin=', 'freqmax='],
            'Fast Fourier Transform' : [0],
            'Hilbert Huang Transform' : [0], #['theta_1=', 'theta_2=', 'alpha=']
            'Wavelets' : [2, 'Haar', 'Daubechies', 'Symlets', 'Coiflets', 'Biorthogonal', 
                            'Reverse biorthogonal', 'Discrete FIR approximation of Meyer wavelet',
                            'Gaussian wavelets', 'Mexican hat wavelet', 'Morlet wavelet',
                            'Complex Gaussian wavelets', 'Shannon wavelets', 'Frequency B-Spline wavelets',
                            'Complex Morlet wavelets' ]
        }
        self.wavelets = {
            'Haar' : ['haar'],
            'Daubechies' : ['db', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
            'Symlets' : ['sym', 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'Coiflets' : ['coif', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            'Biorthogonal' : ['bior', 1.1, 1.3, 1.5, 2.2, 2.4, 2.6, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.4, 5.5, 6.8],
            'Reverse biorthogonal' : ['rbio', 1.1, 1.3, 1.5, 2.2, 2.4, 2.6, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.4, 5.5, 6.8],
            'Discrete FIR approximation of Meyer wavelet' : ['dmey'],
            'Gaussian wavelets' : ['gaus', 1, 2, 3, 4, 5, 6, 7, 8],
            'Mexican hat wavelet' : ['mexh'],
            'Morlet wavelet' : ['morl'],
            'Complex Gaussian wavelets' : ['cgau', 1, 2, 3, 4, 5, 6, 7, 8],
            'Shannon wavelets' : ['shan'],
            'Frequency B-Spline wavelets' : ['fbsp'],
            'Complex Morlet wavelets' : ['cmor']
        }
        self.filter_desc = {
            'Horizontal background removal' : 'Subtracts off row averages for full-width or window-length slices.\n\n:window:\nwindow size - 0 defaults to full length slices',
            'Vertical triangular FIR bandpass' : 'Vertical bandpass filter based on weighted average using a triagular shaped weighting function.\n\n\:freqmin:\nThe lower corner of the bandpass\n:freqmax:\nThe upper corner of the bandpass',
            'Fast Fourier Transform' : 'Converts a signal from the time domain to the frequency domain.',
            'Hilbert Huang Transform' : 'A time series analysis technique which breaks a signal down into Intrinsic Mode Functions (IMFs) which are characterized by being narrowband, nearly monocomponent and having a large time-bandwidth product.\n\n',
                                        # investigate as to what these params do, currently there are default values used by the hht module
                                        # :theta_1: \n\
                                        # Threshold for the stopping criterion\
                                        # :theta_2: \n\
                                        # Threshold for the stopping criterion\
                                        # :alpha: \n\
                                        # Tolerance for the stopping criterion'
            'Wavelets' : 'Computes the Discrete Wavelet Transform using the selected wavelet function\n\nPartial wavlet descriptions at:\nhttp://wavelets.pybytes.com/'
        }
        # check to see if the user actually selected files before attempting to build a tab and read the files
        if (len(self.files_paths) > 0):
            self._build_tab()

    def _build_tab(self):
        print(self.files_paths)
        a_dialog = Alert_Dialog(self)
        a_dialog.show()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(dzt_func, self.files_paths)
            self.orig_data_arrs, self.data_heads = future.result()
        self.filtered_data_arrs = self.orig_data_arrs
        a_dialog.done(0)
        self.setUpdatesEnabled(True)
        # #### Main window layout ####
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        # ==== Setion 1 layout ====
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        # ---- Section 1 label ----
        self.dataLabel = QtWidgets.QLabel(self)
        self.dataLabel.setObjectName("dataLabel")
        self.dataLabel.setText("Data File(s)")
        self.verticalLayout.addWidget(self.dataLabel)
        # ---- List widget for files ----
        self.dataProcessedList = QtWidgets.QListWidget(self)
        self.dataProcessedList.setMinimumWidth(200)
        self.dataProcessedList.setObjectName("dataProcessedList")
        for file in self.files_paths:
            self.dataProcessedList.addItem(os.path.basename(file))
        self.verticalLayout.addWidget(self.dataProcessedList)
        # ==== Section 2 layout ====
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        # ---- Section 2 label ----
        self.appliedFiltersLabel = QtWidgets.QLabel(self)
        self.appliedFiltersLabel.setObjectName("appliedFiltersLabel")
        self.appliedFiltersLabel.setText("Active Filters")
        self.verticalLayout_2.addWidget(self.appliedFiltersLabel)
        # ---- List widget for active filters ----
        self.appliedFilterList = QtWidgets.QListWidget(self)
        self.appliedFilterList.setMinimumWidth(300)
        self.verticalLayout_2.addWidget(self.appliedFilterList)
        # ---- Button to remove selected filter from active filter list ---
        self.clear_filter = QtWidgets.QPushButton('Remove', clicked=lambda: self.remove_filter(self.appliedFilterList.item(self.appliedFilterList.currentRow())))
        self.verticalLayout_2.addWidget(self.clear_filter)
        # ==== Secion 3 layout ====
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        # ---- Section 3 label ----
        self.selectFiltersLabel = QtWidgets.QLabel(self)
        self.selectFiltersLabel.setObjectName("selectFiltersLabel")
        self.selectFiltersLabel.setText("Select Filters")
        self.verticalLayout_3.addWidget(self.selectFiltersLabel)
        # ---- List widget for selecting filters ----
        self.list_filter_holder = QtWidgets.QListWidget()
        self.list_filter_holder.setMinimumWidth(300)
        self.verticalLayout_3.addWidget(self.list_filter_holder)
        for v in self.filter_param_list:
            item = QtWidgets.QListWidgetItem(v)
            self.list_filter_holder.addItem(item)
        self.list_filter_holder.itemClicked.connect(self.get_parameter_inputs)
        # ---- Widget for filter descriptions ----
        self.list_desc = QtWidgets.QLabel()
        self.list_desc.setMinimumWidth(300)
        self.list_desc.setWordWrap(True)
        self.verticalLayout_3.addWidget(self.list_desc)
        # > Divider <
        self.line_2 = QtWidgets.QFrame(self)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.verticalLayout_3.addWidget(self.line_2)
        # ==== Section 4 layout ====
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.horizontalLayout_2.addLayout(self.verticalLayout_4)
        # ---- Widget for parameter inputs ----
        self.list_stack = QtWidgets.QStackedWidget()
        self.list_stack.setMinimumWidth(350)
        self.verticalLayout_4.addWidget(self.list_stack)
        self.list_stack.addWidget(QtWidgets.QLabel(""))
        # > Divider <
        self.line_3 = QtWidgets.QFrame(self)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.verticalLayout_4.addWidget(self.line_3)
        # > Spacer <
        self.verticalLayout_4.addSpacerItem(QtWidgets.QSpacerItem(100, 100, QtWidgets.QSizePolicy.Expanding))
        # ---- Apply button ----
        self.applyButton = QtWidgets.QPushButton(self, clicked=lambda: self.apply_filts())
        self.applyButton.setObjectName("applyButton")    
        self.applyButton.setText("Apply Filters")
        self.verticalLayout_4.addWidget(self.applyButton)
        # ---- Show button ----
        self.showButton = QtWidgets.QPushButton(self, clicked=lambda: self.plot_figure(True))
        self.showButton.setObjectName("showButton")    
        self.showButton.setText("Plot Data")
        self.verticalLayout_4.addWidget(self.showButton)
        # ---- Reset button ----
        self.resetButton = QtWidgets.QPushButton(self, clicked=lambda: self.reset_data())
        self.resetButton.setObjectName("resetButton")    
        self.resetButton.setText("Reset Data")
        self.verticalLayout_4.addWidget(self.resetButton)
        # ---- Export button ----
        self.exportButton = QtWidgets.QPushButton(self)
        self.exportButton.clicked.connect(self.export_pressed)
        self.exportButton.setText("Export Data")
        self.verticalLayout_4.addWidget(self.exportButton)

    def plot_figure(self, show):
        fig, axs = plt.subplots(len(self.filtered_data_arrs), 1, constrained_layout=True)
        # create and array out of the one subpolt that is created if we only have a sinlge input file
        # - avoids errors in for loop code below
        if(len(self.filtered_data_arrs)==1):
            axs = [axs]
        fig.suptitle("DZT Data")
        for i in range(len(self.filtered_data_arrs)):
        # ------ info needed to plot data ---------------------
            mean = np.mean(self.filtered_data_arrs[i])
            std = np.std(self.filtered_data_arrs[i])
            ll = mean - (std * 3)
            ul = mean + (std * 3)
            # ===== Y-AXIS IN DISTANCE UNITS =======
            #   zmax = self.data_heads[i]['rhf_depth'] - self.data_heads[i]['rhf_top']
            #   axs[i].set_ylabel("Depth (m)")
            # ===== Y-AXIS IN TIME UNITS ==========
            zmax = self.data_heads[i]['rhf_range']
            axs[i].set_ylabel('Two-way Time (ns)')
            # ===== X-AXIS IN DISTANCE UNITS ======
            xmax = self.filtered_data_arrs[i].shape[1] / self.data_heads[i]['rhf_spm']
            axs[i].set_xlabel("Distance (m)")
            # ====== X-AXIS IN TIME UNITS =======
            #   xmax = self.data_heads[i]['sec']
            #   axs[i].set_xlabel('Time (s)')
            # =============+=+= SCALING ROUTINE =+=+===============
            # current problem here is that the data is then plotted as very long thin rectangles, not good
            # try:
            #     zscale = self.filtered_data_arrs[i].shape[0]/zmax
            #     xscale = self.filtered_data_arrs[i].shape[1]/xmax
            # except ZeroDivisionError: # apparently this can happen even in genuine GSSI files
            #     zmax = self.filtered_data_arrs[i].shape[0]
            #     zscale = self.filtered_data_arrs[i].shape[0]/zmax
            #     xmax = self.data_heads[i]['sec']
            #     xscale = self.filtered_data_arrs[i].shape[1]/xmax
        # -------------------------------------------------------------------------
            # auto scaling
            axs[i] = axs[i].imshow(self.filtered_data_arrs[i], cmap='gray', clim=(ll, ul), interpolation='bicubic', aspect='auto', extent=[0, xmax, zmax, 2]).axes
            # using the scaling routine above
            # axs[i] = axs[i].imshow(self.filtered_data_arrs[i], cmap='gray', clim=(ll, ul), interpolation='bicubic', aspect=float(zscale)/float(xscale), extent=[0, xmax, zmax, 2]).axes
            axs[i].set_title(os.path.basename(self.files_paths[i]))
        if show:
            plt.show()

    def apply_filts(self):
        a_dialog = Alert_Dialog(self)
        a_dialog.show()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(dzt_filters, self.filtered_data_arrs, self.data_heads, self.active_filters)
            self.filtered_data_arrs = future.result()
        a_dialog.done(0)

    def reset_data(self):
        self.filtered_data_arrs = self.orig_data_arrs
        self.appliedFilterList.clear()

    def remove_filter(self, filt):
        if filt != None:
            self.active_filters.pop(filt.text().split('(')[0])
            self.appliedFilterList.takeItem(self.appliedFilterList.row(self.appliedFilterList.selectedItems()[0]))

    def create_param(self, t, params):
        edit_list = list()
        param_group = QtWidgets.QVBoxLayout()
        if (t == 0):
            return edit_list, param_group
        elif (t == 1):
            for p in params:
                p_group = QtWidgets.QHBoxLayout()
                user_in = QtWidgets.QLineEdit(p.split('=')[1])
                user_in.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("^\d*\.?\d*$"), user_in))
                edit_list.append(user_in)
                p_group.addWidget(QtWidgets.QLabel(p.split('=')[0]))
                p_group.addWidget(user_in)
                param_group.addLayout(p_group)
            return edit_list, param_group
        elif (t == 2):
            p_group = QtWidgets.QListWidget()
            for w in range(len(params)):
                if (len(self.wavelets.get(params[w])) == 1):
                    list_wid = QtWidgets.QListWidgetItem()
                    list_wid.setText(params[w])
                    p_group.addItem(list_wid)
                else:
                    for n in range(1, len(self.wavelets.get(params[w]))):
                        list_wid = QtWidgets.QListWidgetItem()
                        list_wid.setText(params[w] + '_' + str(self.wavelets.get(params[w])[n]))
                        p_group.addItem(list_wid)
            param_group.addWidget(p_group)
            return p_group, param_group

    def get_parameter_inputs(self, item):
        # provide description of selected filter to user
        self.list_desc.setText(item.text()+":\n\n"+self.filter_desc.get(item.text()))
        # create widget for the user to input parameters if applicable
        p = self.filter_param_list.get(item.text())
        if (p[0] == 0):
            # no parameters
            edit_list, param_group = self.create_param(0, p[1:])
        elif (p[0] == 1):
            # parameters are ints/floats
            edit_list, param_group = self.create_param(1, p[1:])
        elif (p[0] == 2):
            # wavelets
            edit_list, param_group = self.create_param(2, p[1:])
        # fill out the user input widgets
        page = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(page)
        box.addWidget(QtWidgets.QLabel(item.text()))
        box.addLayout(param_group)
        apply_button = QtWidgets.QPushButton('Apply', clicked=lambda: self.update_applied_filters_list(item.text(), edit_list))
        cancel_button = QtWidgets.QPushButton('Cancel', clicked=self.on_cancel)
        box.addWidget(apply_button)
        box.addWidget(cancel_button)
        box.addSpacerItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Expanding))
        self.list_stack.removeWidget(self.list_stack.widget(0))
        self.list_stack.insertWidget(0, page)
        self.list_stack.setCurrentIndex(0)
    
    # helper function for paramerter inputs, does a few cleanup actions when cancel button is clicked
    def on_cancel(self):
        self.list_stack.removeWidget(self.list_stack.widget(0))
        self.list_desc.setText("")

    # helper function for paramerter inputs, udates gui and internal variables when user clicks apply button
    def update_applied_filters_list(self, filt, params):
        self.list_desc.setText("")
        p = self.filter_param_list.get(filt)
        new_param = list()
        # to update list we need to find which type of filter it is and update the specified parameters
        if (p[0] == 0):
            # no parameters
            for i in range(self.appliedFilterList.count()):
                if (filt == self.appliedFilterList.item(i).text()):
                    self.appliedFilterList.item(i).setHidden(True)
            self.appliedFilterList.addItem(filt)
            self.active_filters[filt] = True
        elif (p[0] == 1):
            # parameters are ints/floats
            good_inputs = True
            for par in params:
                if (par.text() == ''):
                    good_inputs = False
            if good_inputs:
                # update the internal parameter list
                new_param.append(self.filter_param_list.get(filt)[0])
                for p in range(1, len(self.filter_param_list.get(filt))):
                    new_param.append(self.filter_param_list.get(filt)[p].split('=')[0] + '=' + params[p-1].text())
                self.filter_param_list.update({filt:new_param})
                self.active_filters.update({filt:new_param[1:]})
                # update gui with selected filter
                f_label = filt + '('
                for i in range(len(params)):
                    if i < len(params)-1:
                        f_label = f_label + self.filter_param_list.get(filt)[i+1] + ', '
                    else:
                        f_label = f_label + self.filter_param_list.get(filt)[i+1]
                    f_label = f_label + ')'
                for i in range(self.appliedFilterList.count()):
                    if (filt == self.appliedFilterList.item(i).text().split('(')[0]):
                        self.appliedFilterList.item(i).setHidden(True)
                self.appliedFilterList.addItem(f_label)
        elif (p[0] == 2):
            # selecting a parameter
            if (params.currentItem() is not None):
                # update gui with selected filter
                for i in range(self.appliedFilterList.count()):
                    if (filt == self.appliedFilterList.item(i).text().split('(')[0]):
                        self.appliedFilterList.item(i).setHidden(True)
                wave = params.currentItem().text().split('_')
                code = self.wavelets.get(params.currentItem().text().split('_')[0])[0]
                if (len(wave) == 1):
                    self.appliedFilterList.addItem(filt +'(' + code + ')')
                else:
                    code += params.currentItem().text().split('_')[1]
                    self.appliedFilterList.addItem(filt +'(' + code + ')')
                # update the internal parameter list
                new_param.append(code)
                self.active_filters[filt] = new_param
        self.list_stack.removeWidget(self.list_stack.widget(0))  

    # this method is connected to the export button, will pop up a dialog box and then open up a file explorer box
    def export_pressed(self):
        export_info_box = Export_Dialog(self)
        export_info_box.setWindowTitle("Select Export Options")
        export_info_box.exec_()
        #Determines whether export box was closed by Cancel or Ok
        if export_info_box.result() == 1:
            #Prompts user to select select location to export file using File Explorer
            export_file_placement = str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory"))
            #Checks if no file location was selected to export
            if export_file_placement != "" :
                #Name user selected for file
                output_file_name = export_info_box.file_input_box.text()
                #File format user selects to export as
                output_type = export_info_box.format_selection_box.currentText().lower()
                if output_file_name == "":
                    output_file_name = "output_file"
                    i = 0
                    while os.path.isfile(export_file_placement+output_file_name+output_type):
                            i += 1
                            output_file_name = "output_file" + "(" + str(i) + ")"
                    
                output_abs_path = export_file_placement + "/" + output_file_name
                #Create and show Writing Dialog Modal
                writing_dialog = Writing_Dialog(self)
                writing_dialog.show()
                #Method for writing png output file
                if output_type == "png":
                    self.plot_figure(False)
                    plt.savefig("%s.%s" %(output_abs_path, output_type))
                #Method for writing csv output file
                elif output_type == "csv":
                    if len(self.filtered_data_arrs) == 1:
                        output_abs_path = output_abs_path  + "." + output_type
                        data = pd.DataFrame(self.filtered_data_arrs[0]) # using pandas to output csv
                        data.to_csv(output_abs_path) # write
                    else:
                        for i in range(len(self.filtered_data_arrs)):
                            data = pd.DataFrame(self.filtered_data_arrs[i]) # using pandas to output csv
                            data.to_csv("%s(%d).%s" %(output_abs_path, i+1, output_type))
                #Close Writing Dialog Modal dialog Modal
                writing_dialog.done(0)
#----------------------------------------------------------#
# Parent class widget to manage the dynamic tabs
class tab_manager(QtWidgets.QTabWidget):
    def __init__(self):
        QtWidgets.QTabWidget.__init__(self)
        self.totTabs = 0
        self._build_tabs()

    def _build_tabs(self):
        self.setUpdatesEnabled(True)
        self.insertTab(0, help_tab(),"Home")     
        self.insertTab(1, QtWidgets.QWidget(), '  +  ') 
        self.tabBarClicked.connect(self._add_tab)

    def _add_tab(self, index):
        if index == self.count()-1 :
            # last tab (+) was clicked. add tab
            self.totTabs += 1
            self.insertTab(index, data_tab(), "Tab %d" %(self.totTabs)) 
            self.setCurrentIndex(index)
            if len(self.widget(index).files_paths) == 0:
                self.totTabs -= 1
                self.removeTab(index)
                self.setCurrentIndex(0)
            else:
                self.setCurrentIndex(index)
        
    def _remove_tab(self, index):
        if index == 0 or index == self.count()-1:
            self.setCurrentIndex(0)
        else:
            self.removeTab(index)
            self.setCurrentIndex(0)
#----------------------------------------------------------#
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle("GPR Data Visualizer")
        icon = QtGui.QIcon('favicon.ico')
        MainWindow.setWindowIcon(icon)
        MainWindow.resize(1007, 748)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        tabManager = tab_manager()
        closeTab = QtWidgets.QPushButton('Close Tab', clicked=lambda: tabManager._remove_tab(tabManager.currentIndex()))
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralWidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        
        self.gridLayout_3.addWidget(tabManager, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(closeTab, 1, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1007, 26))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
#----------------------------------------------------------#

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.showMaximized()
    MainWindow.show()
    sys.exit(app.exec_())