import java.awt.Color;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.Graphics;
import java.awt.Toolkit;

import javax.swing.*;


public class Test extends JFrame{

	MojKomponent komponent;
	Siec siec;
	
	public class MojKomponent extends JComponent{

		@Override
		protected void paintComponent(Graphics g) {
			double [] wej=new double[2];
			double [] wynik;
			for(int x=0;x<getWidth();x++)
				for(int y=0;y<getHeight();y++){
					wej[0]=2.0*(x-getWidth()/2)/getWidth();
					wej[1]=2.0*(y-getHeight()/2)/getHeight();
					wynik=siec.oblicz_wyjscie(wej);
					
					/*
					g.setColor(new Color((wynik[0]<0.5)?0:(int)Math.round((wynik[0]-0.5)*510),
							             (wynik[0]>0.5)?0:(int)Math.round((0.5-wynik[0])*510),
							            	 255));
					  */          	 
					g.setColor((wynik[0]>0.5)?Color.RED:Color.GREEN);
					g.fillRect(x, y, 1, 1);	
				}
			super.paintComponent(g);
		}
		
	}
	public Test(String string) {
		super(string);
		setDefaultCloseOperation(EXIT_ON_CLOSE);
		Toolkit kit=Toolkit.getDefaultToolkit();
		Dimension d=kit.getScreenSize();
		setBounds(d.width/4,d.height/4,d.width/2,d.height/2);
		add(komponent=new MojKomponent());
		/*
		 //1 warstwa 1 neuron
		int []tab=new int [1];
		tab[0]=1;
		siec=new Siec(2,1,tab);
		*/
		
		 //3 warstwy
		
		int [] tab=new int [3];
		tab[0]=25; tab[1]=5; tab[2]=1;
		siec=new Siec(2,3,tab);
		
		/*
		 //2 warstwy
		int [] tab=new int [2];
		tab[0]=10; tab[1]=1;
		siec=new Siec(2,2,tab);
		*/
		setVisible(true);
	}

	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {
			
			@Override
			public void run() {
				new Test("neurony");
			}
		});
	}

}
