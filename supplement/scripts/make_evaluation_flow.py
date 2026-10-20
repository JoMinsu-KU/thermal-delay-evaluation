"""Vector Figure 1: monochrome execution flow with attached feedback routing."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path as MplPath

def draw_evaluation_flow(output_dir):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':7,
                         'pdf.fonttype':42,'svg.fonttype':'none'}):
        # Coordinates are in inches, so the same arrow proportions survive export.
        fig=plt.figure(figsize=(3.15,3.58),facecolor='white')
        ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,3.15),ylim=(0,3.58));ax.axis('off')
        left,right=.47,3.10;center=(left+right)/2;h=.47
        nodes=[
            (3.25,'Observation and command history','Causal sensors, frames, and process context'),
            (2.63,'Nine candidate heater commands','Hold, or change one heater by +/-0.04'),
            (2.01,'Candidate scoring and\nPINO feasibility gate','Method-specific scores; zero-command fallback'),
            (1.39,'Asynchronous command execution','Path-matched delay; 100-ms deadline\nOn expiration: zero or hold command'),
            (.77,'Finite-volume thermal plant','60 s trajectory; state evolves during delay'),
        ]
        for y,title,detail in nodes:
            ax.add_patch(Rectangle((left,y-h/2),right-left,h,
                                  facecolor='#f2f2f2' if y==1.39 else 'white',
                                  edgecolor='#202020',linewidth=.8,zorder=2))
            ax.text(center,y+(.06 if y==2.01 else .105),title,ha='center',va='center',fontsize=7.7,
                    linespacing=1.03,
                    fontweight='bold',color='black',zorder=3)
            ax.text(center,y-(.155 if y==2.01 else .070),detail,ha='center',va='center',fontsize=6.75,
                    linespacing=1.25,color='black',zorder=3)
        arrow=dict(arrowstyle='-|>',mutation_scale=8,linewidth=.8,
                   edgecolor='#202020',facecolor='#202020',shrinkA=0,shrinkB=0,
                   capstyle='butt',joinstyle='miter',zorder=1)
        for (upper,*_),(lower,*__) in zip(nodes,nodes[1:]):
            ax.add_patch(FancyArrowPatch((center,upper-h/2),
                                        (center,lower+h/2),**arrow))
        # Return from plant to observation; the arrowhead enters the first box.
        loop_x=.19;low=nodes[-1][0];high=nodes[0][0]
        route=MplPath([(left,low),(loop_x,low),(loop_x,high),(left,high)],
                      [MplPath.MOVETO,MplPath.LINETO,MplPath.LINETO,MplPath.LINETO])
        ax.add_patch(FancyArrowPatch(path=route,**arrow))
        ax.text(.075,(low+high)/2,'State feedback',rotation=90,ha='center',
                va='center',fontsize=6.75,color='#303030')
        ax.text(center,.345,'Evaluation readouts',ha='center',va='center',
                fontsize=7.1,fontweight='bold',color='black')
        ax.text(center,.165,'FV cost difference / trajectory cost / events',
                ha='center',va='center',fontsize=6.75,color='black')
        # Export with a fixed canvas; PDF stays vector and fonts remain embedded.
        fig.savefig(output_dir/'evaluation_flow.pdf',
                    metadata={'Author':'','Title':'Candidate evaluation and asynchronous execution',
                              'CreationDate':None,'ModDate':None})
        fig.savefig(output_dir/'evaluation_flow.png',dpi=300)
        fig.savefig(output_dir/'evaluation_flow.svg',metadata={'Date':None,'Creator':'Matplotlib'})
        plt.close(fig)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'figures')
    draw_evaluation_flow(parser.parse_args().output)
