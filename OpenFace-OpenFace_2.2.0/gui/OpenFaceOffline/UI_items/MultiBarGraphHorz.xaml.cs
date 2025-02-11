﻿///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// THIS SOFTWARE IS PROVIDED “AS IS” FOR ACADEMIC USE ONLY AND ANY EXPRESS
// OR IMPLIED WARRANTIES WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY.
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Notwithstanding the license granted herein, Licensee acknowledges that certain components
// of the Software may be covered by so-called “open source” software licenses (“Open Source
// Components”), which means any software licenses approved as open source licenses by the
// Open Source Initiative or any substantially similar licenses, including without limitation any
// license that, as a condition of distribution of the software licensed under such license,
// requires that the distributor make the software available in source code format. Licensor shall
// provide a list of Open Source Components for a particular version of the Software upon
// Licensee’s request. Licensee will comply with the applicable terms of such licenses and to
// the extent required by the licenses covering Open Source Components, the terms of such
// licenses will apply in lieu of the terms of this Agreement. To the extent the terms of the
// licenses applicable to Open Source Components prohibit any of the restrictions in this
// License Agreement with respect to such Open Source Component, such restrictions will not
// apply to such Open Source Component. To the extent the terms of the licenses applicable to
// Open Source Components require Licensor to make an offer to provide source code or
// related information in connection with the Software, such offer is hereby made. Any request
// for source code or related information should be directed to cl-face-tracker-distribution@lists.cam.ac.uk
// Licensee acknowledges receipt of notices for the Open Source Components for the initial
// delivery of the Software.

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace OpenFaceOffline
{
    /// <summary>
    /// Interaction logic for MultiBarGraphHorz.xaml
    /// </summary>
    public partial class MultiBarGraphHorz : UserControl
    {
        int num_bars = 0;
        Dictionary<String, BarGraphHorizontal> graphs;

        // Name mapping
        Dictionary<String, String> mapping;

        public MultiBarGraphHorz()
        {
            InitializeComponent();

            graphs = new Dictionary<string, BarGraphHorizontal>();

            mapping = new Dictionary<string, string>();
            mapping["AU01"] = "Inner Brow raiser";
            mapping["AU02"] = "Outer Brow raiser";
            mapping["AU04"] = "Brow lowerer";
            mapping["AU05"] = "Upper lid raiser";
            mapping["AU06"] = "Cheek raiser";
            mapping["AU07"] = "Lid tightener";
            mapping["AU09"] = "Nose wrinkler";
            mapping["AU10"] = "Upper lip raiser";
            mapping["AU12"] = "Lip corner puller (smile)";
            mapping["AU14"] = "Dimpler";
            mapping["AU15"] = "Lip corner depressor";
            mapping["AU17"] = "Chin Raiser";
            mapping["AU20"] = "Lip Stretcher";
            mapping["AU23"] = "Lip tightener";
            mapping["AU25"] = "Lips part";
            mapping["AU26"] = "Jaw drop";
            mapping["AU28"] = "Lip suck";
            mapping["AU45"] = "Blink";



        }

        public void Update(Dictionary<String, double> data)
        {
            // Create new bars if necessary
            if (num_bars != data.Count)
            {
                num_bars = data.Count;
                barGrid.Children.Clear();
                barGrid.RowDefinitions.Clear();
                graphs.Clear();

                // Make sure AUs are sorted
                var data_labels = data.Keys.ToList();
                data_labels.Sort();

                foreach (var label in data_labels)
                {
                    BarGraphHorizontal newBar = new BarGraphHorizontal(label + " - " + mapping[label]);
                    newBar.SetValue(data[label]);
                    barGrid.RowDefinitions.Add(new RowDefinition());
                    Grid.SetRow(newBar, graphs.Count);
                    graphs.Add(label, newBar);
                    barGrid.Children.Add(newBar);
                }

            }

            // Update the bars
            foreach (var value in data)
            {
                graphs[value.Key].SetValue(value.Value);
            }            
        }
    }
}
